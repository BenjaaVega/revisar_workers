from celery import shared_task
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from psycopg2.extras import RealDictCursor
import os, time, math
from typing import Dict, List, Any, Tuple
from threading import RLock
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

try:
    CACHE_TTL = max(int(os.getenv("PROPERTIES_CACHE_TTL", "300")), 0)
except ValueError:
    CACHE_TTL = 300

FEATURE_COLUMNS = [
    'price',
    'bedrooms',
    'bathrooms',
    'm2',
    'price_per_m2',
    'room_ratio',
    'total_rooms',
    'location_cluster'
]

_properties_cache: Dict[str, Any] = {"data": None, "expires_at": 0.0}
_preprocessed_cache: Dict[str, Any] = {"df": None, "expires_at": 0.0}
_cache_lock = RLock()

# ---------- DB ----------
def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        cursor_factory=RealDictCursor,
        connect_timeout=10
    )

def _fetch_properties_from_db() -> List[Dict[str, Any]]:
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, name, price, bedrooms, bathrooms, m2, location,
                   is_project, visit_slots, timestamp,
                   comuna, lat, lon
            FROM properties
            WHERE price IS NOT NULL AND price > 0
              AND m2 IS NOT NULL AND m2 > 0
            ORDER BY timestamp DESC
        """)
        props = cur.fetchall()
        return [dict(p) for p in props]
    finally:
        cur.close(); conn.close()


def get_properties_data(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Obtain raw properties data with a small in-process cache.

    A lightweight cache avoids re-reading the full table on each task execution,
    which previously caused high latency in the workers when several jobs were
    queued simultaneously.
    """
    now = time.time()
    if not force_refresh and CACHE_TTL and now < _properties_cache["expires_at"] and _properties_cache["data"]:
        return _properties_cache["data"]

    data = _fetch_properties_from_db()

    if CACHE_TTL:
        with _cache_lock:
            _properties_cache["data"] = data
            _properties_cache["expires_at"] = now + CACHE_TTL

    return list(data)


def get_preprocessed_properties(force_refresh: bool = False) -> pd.DataFrame:
    now = time.time()
    if not force_refresh and CACHE_TTL and now < _preprocessed_cache["expires_at"] and _preprocessed_cache["df"] is not None:
        return _preprocessed_cache["df"]

    properties = get_properties_data(force_refresh=force_refresh)
    df = preprocess_properties(properties) if properties else pd.DataFrame()

    if CACHE_TTL:
        with _cache_lock:
            _preprocessed_cache["df"] = df
            _preprocessed_cache["expires_at"] = now + CACHE_TTL

    return df


def preprocess_properties(properties: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(properties)

    
    def loc_address(x):
        if isinstance(x, dict):
            return x.get('address', '')
        return str(x) if x is not None else ''
    df['location_address'] = df['location'].apply(loc_address)

    # Sanitizar valores (quitar scrap malos)
    df = df[(df['price'] >= 100000) & (df['price'] <= 10_000_000_000)]
    df = df[(df['m2'] >= 15) & (df['m2'] <= 1000)]

    
    df['location_cluster'] = df['location_address'].apply(lambda s: hash(s.lower()) % 10)

    
    df['price_per_m2'] = df['price'] / df['m2']
    df['room_ratio']    = df['bedrooms'] / (df['bathrooms'] + 1)  # evitar div/0
    df['total_rooms']   = df['bedrooms'] + df['bathrooms']

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean(numeric_only=True))

    return df[FEATURE_COLUMNS + ['id','name','location_address','comuna','lat','lon']]

def apply_clustering_algorithm(df: pd.DataFrame, n_clusters: int = 8) -> Tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df['cluster'] = clusters
    return df, scaler, X_scaled

def calculate_user_preference_vector(user_preferences: Dict[str, Any], defaults: Dict[str, float]) -> np.ndarray:
    vec = np.array([defaults.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=float)

    budget_min = user_preferences.get('budget_min')
    budget_max = user_preferences.get('budget_max')
    if budget_min is not None or budget_max is not None:
        values = [v for v in (budget_min, budget_max) if v is not None]
        if values:
            avg_budget = sum(values) / len(values)
            vec[0] = float(avg_budget)

    bedrooms = user_preferences.get('bedrooms')
    if bedrooms is not None:
        try:
            bedrooms = float(bedrooms)
            vec[1] = bedrooms
        except (TypeError, ValueError):
            pass

    bathrooms = user_preferences.get('bathrooms')
    if bathrooms is not None:
        try:
            bathrooms = float(bathrooms)
            vec[2] = bathrooms
        except (TypeError, ValueError):
            pass

    if bedrooms is not None and bathrooms is not None:
        try:
            vec[FEATURE_COLUMNS.index('room_ratio')] = bedrooms / (bathrooms + 1)
            vec[FEATURE_COLUMNS.index('total_rooms')] = bedrooms + bathrooms
        except ZeroDivisionError:
            pass

    if user_preferences.get('location'):
        vec[FEATURE_COLUMNS.index('location_cluster')] = (hash(str(user_preferences['location']).lower()) % 10)

    return vec


def find_similar_properties(
    df: pd.DataFrame,
    user_preferences: Dict[str, Any],
    n_recommendations: int = 10,
    scaler: StandardScaler = None,
    scaled_features: np.ndarray = None,
    defaults: Dict[str, float] = None
) -> List[Dict]:
    if defaults is None:
        defaults = df[FEATURE_COLUMNS].mean(numeric_only=True).to_dict()

    if scaler is None or scaled_features is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[FEATURE_COLUMNS])

    user_vector = calculate_user_preference_vector(user_preferences, defaults)
    user_vector_scaled = scaler.transform([user_vector])
    similarities = cosine_similarity(user_vector_scaled, scaled_features)[0]

    df = df.copy()
    df['similarity_score'] = similarities

    # Filtros  por preferencias
    filtered = df
    if user_preferences.get('budget_min') is not None:
        filtered = filtered[filtered['price'] >= user_preferences['budget_min']]
    if user_preferences.get('budget_max') is not None:
        filtered = filtered[filtered['price'] <= user_preferences['budget_max']]
    if user_preferences.get('bedrooms') is not None:
        filtered = filtered[filtered['bedrooms'] >= int(user_preferences['bedrooms'])]
    if user_preferences.get('bathrooms') is not None:
        filtered = filtered[filtered['bathrooms'] >= int(user_preferences['bathrooms'])]
    if user_preferences.get('location'):
        loc = str(user_preferences['location']).lower()
        filtered = filtered[filtered['location_address'].str.lower().str.contains(loc, na=False)]

    recs = filtered.nlargest(n_recommendations, 'similarity_score')
    out = []
    for r in recs.itertuples(index=False):
        out.append({
            "property_id": str(r.id),
            "name": r.name,
            "price": float(r.price),
            "bedrooms": int(r.bedrooms),
            "bathrooms": int(r.bathrooms),
            "m2": float(r.m2),
            "location": r.location_address,
            "similarity_score": float(r.similarity_score),
            "cluster": int(r.cluster),
            "price_per_m2": float(r.price_per_m2),
            "match_reason": f"High similarity score ({float(r.similarity_score):.2f}) and matches your preferences"
        })
    return out

# clustering 
@shared_task(name="tasks.generate_recommendations", bind=True)
def generate_recommendations(self, job_id: str, user_id: str, preferences: dict):
    t0 = time.perf_counter()
    self.update_state(state="PROGRESS", meta={"progress": 5})

    df = get_preprocessed_properties()
    if df.empty:
        return {"recommendations": [], "total_found": 0, "error": "No properties found in database"}

    self.update_state(state="PROGRESS", meta={"progress": 30})
    df, scaler, scaled_features = apply_clustering_algorithm(df, n_clusters=8)

    self.update_state(state="PROGRESS", meta={"progress": 80})
    defaults = df[FEATURE_COLUMNS].mean(numeric_only=True).to_dict()
    recs = find_similar_properties(
        df,
        preferences or {},
        n_recommendations=10,
        scaler=scaler,
        scaled_features=scaled_features,
        defaults=defaults
    )

    dt = time.perf_counter() - t0
    self.update_state(state="PROGRESS", meta={"progress": 100})

    return {
        "recommendations": recs,
        "total_found": len(recs),
        "algorithm_used": "k-means_clustering_with_cosine_similarity",
        "clusters_used": 8,
        "processing_time": f"{dt:.2f}s",
        "user_id": user_id,
        "job_id": job_id
    }

# Tarea: regla del enunciado 
@shared_task(name="tasks.generate_recommendations_simple", bind=True)
def generate_recommendations_simple(self, property_id: int):
    t0 = time.perf_counter()
    self.update_state(state="PROGRESS", meta={"progress": 5})

    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT id, comuna, bedrooms, price, lat, lon FROM properties WHERE id=%s", (property_id,))
        base = cur.fetchone()
        if not base:
            return {"recommendations": [], "total_found": 0, "reason": "base_not_found"}

        self.update_state(state="PROGRESS", meta={"progress": 20})

        cur.execute("""
            SELECT id, name, price, bedrooms, bathrooms, m2, lat, lon, url
            FROM properties
            WHERE comuna = %s
              AND bedrooms = %s
              AND price <= %s
              AND id <> %s
        """, (base["comuna"], base["bedrooms"], base["price"], base["id"]))
        cands = cur.fetchall()

        self.update_state(state="PROGRESS", meta={"progress": 60})

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000.0
            dphi = math.radians((lat2 or 0) - (lat1 or 0))
            dlmb = math.radians((lon2 or 0) - (lon1 or 0))
            a = (math.sin(dphi/2)**2 +
                 math.cos(math.radians(lat1 or 0))*math.cos(math.radians(lat2 or 0))*math.sin(dlmb/2)**2)
            return 2*R*math.asin(math.sqrt(a))

        def sort_key(r):
            if not base["lat"] or not base["lon"] or not r["lat"] or not r["lon"]:
                return (float("inf"), float(r["price"]) if r["price"] is not None else 1e18)
            d = haversine(base["lat"], base["lon"], r["lat"], r["lon"])
            return (d, float(r["price"]) if r["price"] is not None else 1e18)

        top = sorted(cands, key=sort_key)[:3]
        dt = time.perf_counter() - t0

        out = []
        for r in top:
            out.append({
                "property_id": str(r["id"]),
                "name": r["name"],
                "price": float(r["price"]) if r["price"] is not None else None,
                "bedrooms": int(r["bedrooms"]) if r["bedrooms"] is not None else None,
                "bathrooms": int(r["bathrooms"]) if r["bathrooms"] is not None else None,
                "m2": float(r["m2"]) if r["m2"] is not None else None,
                "url": r["url"]
            })

        self.update_state(state="PROGRESS", meta={"progress": 100})

        return {
            "recommendations": out,
            "total_found": len(out),
            "reason": "ok",
            "processing_time": f"{dt:.2f}s"
        }
    finally:
        cur.close(); conn.close()
