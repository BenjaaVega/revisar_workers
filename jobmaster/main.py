from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from celery import Celery
from celery.result import AsyncResult
import os, json, redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Redis para almacenar jobs (TTL 24h)
redis_client = redis.from_url(REDIS_URL)

def store_job(job_id: str, data: dict, ttl: int = 86400):
    redis_client.setex(f"job:{job_id}", ttl, json.dumps(data))

def get_job(job_id: str):
    raw = redis_client.get(f"job:{job_id}")
    return json.loads(raw) if raw else None

# Celery (misma URL que workers)
cel = Celery("recommendation_worker", broker=REDIS_URL, backend=REDIS_URL)

app = FastAPI(title="JobMaster Service", version="1.1.0")


class JobRequest(BaseModel):
    user_id: str
    property_id: Optional[int] = None
    preferences: Optional[Dict[str, Any]] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    location: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None

# Endpoints

@app.get("/")
def root():
    return {
        "service": "JobMaster",
        "version": "1.1.0",
        "endpoints": {
            "create_job": "POST /job",
            "get_job": "GET /job/{job_id}",
            "list_jobs": "GET /jobs",
            "heartbeat": "GET /heartbeat"
        }
    }

@app.get("/heartbeat")
def heartbeat():
    ok = True
    workers_count = 0
    try:
        redis_client.ping()
        pings = cel.control.ping() or []
        workers_count = len(pings)
    except Exception:
        ok = False
    return {
        "status": ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "JobMaster",
        "workers_active": workers_count
    }

@app.post("/job")
def create_job(job: JobRequest):
    import uuid
    job_id = str(uuid.uuid4())
    data = {
        "job_id": job_id,
        "user_id": job.user_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preferences": job.preferences or {},
        "property_id": job.property_id,
        "budget_min": job.budget_min,
        "budget_max": job.budget_max,
        "location": job.location,
        "bedrooms": job.bedrooms,
        "bathrooms": job.bathrooms
    }
    store_job(job_id, data)

    # Switch: si llega property_id -> usa regla del enunciado; si no, clustering
    if job.property_id is not None:
        async_res = cel.send_task("tasks.generate_recommendations_simple", args=[int(job.property_id)])
    else:
        async_res = cel.send_task("tasks.generate_recommendations",
                                  args=[job_id, job.user_id, job.preferences or {}])

    if not async_res or not async_res.id:
        data["status"] = "failed"
        data["error"] = "task_not_scheduled"
        store_job(job_id, data)
        raise HTTPException(status_code=500, detail="Celery did not return a task id")

    data["task_id"] = async_res.id
    store_job(job_id, data)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job created successfully",
        "created_at": data["created_at"]
    }

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    data = get_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    task_id = data.get("task_id")
    if not task_id:
        # evita AsyncResult(None)
        return {
            "job_id": data["job_id"],
            "status": data["status"],
            "result": data.get("result"),
            "error": data.get("error") or "task_id_missing",
            "created_at": data["created_at"],
            "completed_at": data.get("completed_at"),
            "progress": data.get("progress")
        }

    r = AsyncResult(task_id, app=cel)
    state = r.state 

    if state == "SUCCESS":
        data["status"] = "completed"
        data["result"] = r.result
     
        try:
            if data.get("result") and data["result"].get("recommendations"):
                data["result"]["recommendations"] = data["result"]["recommendations"][:3]
        except Exception:
            pass
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        store_job(job_id, data)

    elif state == "FAILURE":
        data["status"] = "failed"
        data["error"] = str(r.info)
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        store_job(job_id, data)

    elif state == "PROGRESS":
        info = r.info or {}
        data["status"] = "processing"
        data["progress"] = info.get("progress", 0)
        store_job(job_id, data)

    else:
        data["status"] = state.lower()
        store_job(job_id, data)

    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "created_at": data["created_at"],
        "completed_at": data.get("completed_at"),
        "progress": data.get("progress")
    }

@app.get("/jobs")
def list_jobs(user_id: Optional[str] = None, status: Optional[str] = None):
    keys = redis_client.keys("job:*")
    jobs = []
    for k in keys:
        try:
            d = json.loads(redis_client.get(k))
            if user_id and d.get("user_id") != user_id:
                continue
            if status and d.get("status") != status:
                continue
            jobs.append({
                "job_id": d["job_id"],
                "user_id": d.get("user_id"),
                "status": d.get("status"),
                "created_at": d.get("created_at"),
                "completed_at": d.get("completed_at")
            })
        except Exception:
            continue
    jobs.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return {"jobs": jobs, "total": len(jobs)}
