from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

app = Celery(
    "recommendation_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,      # 30 min
    task_soft_time_limit=25 * 60, # 25 min
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Importa el módulo de tareas para registrarlas
import tasks  
