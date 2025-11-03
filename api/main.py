from fastapi.responses import StreamingResponse
import io
try:
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None

import os
from fastapi import FastAPI, HTTPException, Query, Response, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid
from pydantic import BaseModel
import paho.mqtt.client as mqtt
import uuid as uuidlib
from time import sleep
import json
from webpay_service import WebPayService
import requests

# Importar la dependencia de autenticación
from auth import verify_jwt
from webpay_service import WebPayService
from email_service import EmailService

# Crear instancia del servicio WebPay
webpay_service = WebPayService()

# Crear instancia del servicio de Email
email_service = EmailService()

# Cargar variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
INSTANCE_NAME = os.getenv("CONTAINER_NAME", "fastapi_unknown")

MQTT_BROKER = os.getenv("BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
REQUESTS_TOPIC = os.getenv("REQUESTS_TOPIC", "properties/requests")
VALIDATION_TOPIC = os.getenv("VALIDATION_TOPIC", "properties/validation")
GROUP_ID = os.getenv("GROUP_ID", "gX")

# Worker service configuration
WORKER_SERVICE_URL = os.getenv("WORKER_SERVICE_URL", "http://localhost:8003")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://iic2173-e0-repablo6.me")


class VisitRequestIn(BaseModel):
    url: str

class VisitRequestOut(BaseModel):
    request_id: str
    status: str
    message: str

class MyProperty(BaseModel):
    request_id: str
    url: str
    status: str
    created_at: str
    amount: float
    has_receipt: bool
    property: dict

class PurchaseDetail(BaseModel):
    request_id: str
    url: str
    status: str
    created_at: str
    amount: float
    has_receipt: bool
    property: dict
    rejection_reason: Optional[str] = None
    authorization_code: Optional[str] = None

def mqtt_publish_with_fibonacci(topic: str, payload: str, max_retries: int = 6):
    fib = [1, 1]
    for _ in range(max_retries - 2):
        fib.append(fib[-1] + fib[-2])

    attempt = 0
    while True:
        try:
            client = mqtt.Client()
            if MQTT_USER and MQTT_PASSWORD:
                client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_start()
            res = client.publish(topic, payload, qos=1)
            res.wait_for_publish()
            client.loop_stop()
            client.disconnect()
            return True
        except Exception:
            if attempt >= len(fib) - 1:
                return False
            sleep(fib[attempt])
            attempt += 1

def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        cursor_factory=RealDictCursor,
        connect_timeout=10,  # Timeout de conexión de 10 segundos
        application_name="fastapi_app"
    )

def ensure_user_exists(user_id: str, name: str, email: str, phone: str = None):
    """Crear usuario si no existe, NO actualizar si existe"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Intentar obtener usuario existente
        cur.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
        user = cur.fetchone()
        
        if not user:
            # Usuario no existe, crear nuevo
            cur.execute(
                "INSERT INTO users (user_id, name, email, phone) VALUES (%s, %s, %s, %s)",
                (user_id, name, email, phone)
            )
            # Crear wallet para el usuario
            cur.execute(
                "INSERT INTO wallets (user_id, balance) VALUES (%s, 0.00)",
                (user_id,)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def update_user_data(user_id: str, name: str, email: str, phone: str = None):
    """Actualizar datos del usuario existente"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "UPDATE users SET name = %s, email = %s, phone = %s, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
            (name, email, phone, user_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def get_user_balance(user_id: str) -> float:
    """Obtener saldo del usuario"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT balance FROM wallets WHERE user_id = %s", (user_id,))
        result = cur.fetchone()
        return float(result['balance']) if result else 0.0
    finally:
        cur.close()
        conn.close()

def update_wallet_balance(user_id: str, new_balance: float):
    """Actualizar saldo del wallet"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "UPDATE wallets SET balance = %s, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
            (new_balance, user_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def create_transaction(user_id: str, transaction_type: str, amount: float, description: str, property_id: str = None) -> str:
    """Crear transacción y retornar ID"""
    transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "INSERT INTO transactions (id, user_id, type, amount, description, property_id) VALUES (%s, %s, %s, %s, %s, %s)",
            (transaction_id, user_id, transaction_type, amount, description, property_id)
        )
        conn.commit()
        return transaction_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


app = FastAPI(title="API de Propiedades")

# Configuración de CORS para permitir el frontend
origins = [
    os.getenv("FRONTEND_ORIGIN", "https://iic2173-e0-repablo6.me"),
    "https://www.iic2173-e0-repablo6.me",
    "https://dbdcin4y3ybd.cloudfront.net",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Endpoint adicional para manejar preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


# Modelos Pydantic
class UserUpdate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class UserResponse(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    user_id: str

class WalletResponse(BaseModel):
    balance: float
    user_id: str

class DepositRequest(BaseModel):
    amount: float

class DepositResponse(BaseModel):
    new_balance: float
    transaction_id: str
    message: str

class PurchaseRequest(BaseModel):
    property_id: str
    amount: float

class PurchaseResponse(BaseModel):
    new_balance: float
    transaction_id: str
    message: str

class PurchaseErrorResponse(BaseModel):
    error: str
    current_balance: float
    required_amount: float

class TransactionResponse(BaseModel):
    id: str
    type: str
    amount: float
    created_at: str
    description: str

# Modelos para WebPay
class WebPayCreateRequest(BaseModel):
    amount: float
    url: str  # URL de la propiedad para reservar
    description: Optional[str] = "Reserva de visita"

class WebPayCreateResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None

class WebPayCommitRequest(BaseModel):
    token: str
    url: str  # URL de la propiedad para la cual se validó el pago

class WebPayCommitResponse(BaseModel):
    success: bool
    request_id: Optional[str] = None
    message: Optional[str] = None
    transaction: Optional[dict] = None
    error: Optional[str] = None

# Worker service models
class RecommendationRequest(BaseModel):
    property_id: Optional[str] = None
    preferences: Optional[dict] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    location: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None

class RecommendationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str

class WorkerHeartbeatResponse(BaseModel):
    status: bool
    timestamp: str
    service: str
    workers_active: int

@app.get("/properties")
def list_properties(
    response: Response,
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1),
    price: Optional[float] = None,
    location: Optional[str] = None,
    date: Optional[str] = None
):
    response.headers["X-Instance-Name"] = INSTANCE_NAME

    offset = (page - 1) * limit
    query = """
        SELECT
            id,
            name,
            price,
            currency,
            bedrooms,
            bathrooms,
            m2,
            location,
            img,
            url,
            is_project,
            visit_slots,
            timestamp AS last_updated
        FROM properties
        WHERE 1=1
    """
#Como cada mensaje properties/info lo insertamos como una fila con visit_slots=1,
#  y al reservar restamos visit_slots -= 1 por url, por el agregado correcto es SUM(visit_slots)

    params = []

    if price is not None:
        query += " AND price <= %s"
        params.append(price)
    if location:
        query += " AND LOWER(location->>'address') LIKE %s"
        params.append(f"%{location.lower()}%")
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            query += " AND DATE(timestamp) = %s"
            params.append(dt.date())
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha inválido, usar YYYY-MM-DD")

    query += """
        ORDER BY last_updated DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, tuple(params))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

@app.get("/properties/{property_id}")
def get_property(property_id: int, response: Response, user: dict = Depends(verify_jwt)):
    response.headers["X-Instance-Name"] = INSTANCE_NAME

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM properties WHERE id=%s", (property_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        raise HTTPException(status_code=404, detail="Propiedad no encontrada")
    return result


# ===== ENDPOINTS DE USUARIO =====

@app.get("/me", response_model=UserResponse)
def get_user_profile(user: dict = Depends(verify_jwt)):
    """Obtener datos del usuario loggeado"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Leer datos desde la BD
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "SELECT name, email, phone FROM users WHERE user_id = %s",
            (user_id,)
        )
        user_data = cur.fetchone()
        
        if user_data:
            # Usuario existe en BD, devolver datos de BD
            return UserResponse(
                name=user_data['name'],
                email=user_data['email'],
                phone=user_data['phone'],
                user_id=user_id
            )
        else:
            # Usuario no existe en BD, crear con datos del token
            ensure_user_exists(user_id, name, email, phone)
            return UserResponse(
                name=name,
                email=email,
                phone=phone,
                user_id=user_id
            )
    finally:
        cur.close()
        conn.close()

@app.put("/me", response_model=UserResponse)
def update_user_profile(user_data: UserUpdate, user: dict = Depends(verify_jwt)):
    """Actualizar datos de contacto del usuario"""
    user_id = user.get("sub")
    
    # Validar datos
    if not user_data.name or not user_data.email:
        raise HTTPException(status_code=400, detail="Nombre y email son requeridos")
    
    # Asegurar que el usuario existe
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    ensure_user_exists(user_id, name, email, phone)
    
    # Actualizar datos del usuario
    update_user_data(user_id, user_data.name, user_data.email, user_data.phone)
    
    return UserResponse(
        name=user_data.name,
        email=user_data.email,
        phone=user_data.phone,
        user_id=user_id
    )

# ===== ENDPOINTS DE WALLET =====

@app.get("/wallet", response_model=WalletResponse)
def get_wallet_balance(user: dict = Depends(verify_jwt)):
    """Obtener saldo actual del usuario"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Asegurar que el usuario existe
    ensure_user_exists(user_id, name, email, phone)
    
    balance = get_user_balance(user_id)
    
    return WalletResponse(
        balance=balance,
        user_id=user_id
    )

@app.post("/wallet/deposit", response_model=DepositResponse)
def deposit_to_wallet(deposit_data: DepositRequest, user: dict = Depends(verify_jwt)):
    """Cargar dinero al wallet"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Validar monto
    if deposit_data.amount <= 0:
        raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0")
    
    # Asegurar que el usuario existe
    ensure_user_exists(user_id, name, email, phone)
    
    # Obtener saldo actual
    current_balance = get_user_balance(user_id)
    new_balance = current_balance + deposit_data.amount
    
    # Actualizar saldo
    update_wallet_balance(user_id, new_balance)
    
    # Crear transacción
    transaction_id = create_transaction(
        user_id=user_id,
        transaction_type="deposit",
        amount=deposit_data.amount,
        description="Carga de wallet"
    )
    
    return DepositResponse(
        new_balance=new_balance,
        transaction_id=transaction_id,
        message="Depósito exitoso"
    )

@app.get("/wallet/transactions", response_model=list[TransactionResponse])
def get_wallet_transactions(user: dict = Depends(verify_jwt)):
    """Obtener historial de transacciones"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Asegurar que el usuario existe
    ensure_user_exists(user_id, name, email, phone)
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "SELECT id, type, amount, description, created_at FROM transactions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        transactions = cur.fetchall()
        
        return [
            TransactionResponse(
                id=tx['id'],
                type=tx['type'],
                amount=float(tx['amount']),
                created_at=tx['created_at'].isoformat() + "Z",
                description=tx['description']
            )
            for tx in transactions
        ]
    finally:
        cur.close()
        conn.close()

@app.post("/wallet/purchase")
def purchase_property(purchase_data: PurchaseRequest, user: dict = Depends(verify_jwt)):
    """Procesar compra de propiedad"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Validar monto
    if purchase_data.amount <= 0:
        raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0")
    
    # Asegurar que el usuario existe
    ensure_user_exists(user_id, name, email, phone)
    
    # Obtener saldo actual
    current_balance = get_user_balance(user_id)
    
    # Verificar saldo suficiente
    if current_balance < purchase_data.amount:
        return PurchaseErrorResponse(
            error="Saldo insuficiente",
            current_balance=current_balance,
            required_amount=purchase_data.amount
        )
    
    # Calcular nuevo saldo
    new_balance = current_balance - purchase_data.amount
    
    # Actualizar saldo
    update_wallet_balance(user_id, new_balance)
    
    # Crear transacción
    transaction_id = create_transaction(
        user_id=user_id,
        transaction_type="purchase",
        amount=purchase_data.amount,
        description="Compra de propiedad",
        property_id=purchase_data.property_id
    )
    
    return PurchaseResponse(
        new_balance=new_balance,
        transaction_id=transaction_id,
        message="Compra realizada exitosamente"
    )

@app.api_route("/", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def root():
    """Endpoint raíz para API Gateway - acepta todos los métodos HTTP"""
    return {
        "message": "API funcionando correctamente",
        "status": "healthy",
        "instance": INSTANCE_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Endpoint de salud sin autenticación"""
    return {
        "status": "healthy",
        "instance": INSTANCE_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/db")
def health_check_db():
    """Verificar conectividad a la base de datos"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/auth/test")
def auth_test(user: dict = Depends(verify_jwt)):
    """Endpoint simple para probar autenticación sin BD"""
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    return {
        "status": "authenticated",
        "user_id": user.get("sub"),
        "name": user.get("name"),
        "email": email,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/visits/request", response_model=VisitRequestOut)
def create_visit_request(data: VisitRequestIn, user: dict = Depends(verify_jwt)):
    """
    RF05: Publica una solicitud de compra en properties/requests y registra en BD como PENDING.
    NO descuenta saldo aún; el descuento ocurre solo si llega VALIDATION con ACCEPTED.
    """
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")

    ensure_user_exists(user_id, name, email, phone)

    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT price, currency, visit_slots FROM properties WHERE url = %s ORDER BY timestamp DESC LIMIT 1", (data.url,))
        prop = cur.fetchone()
        if not prop:
            raise HTTPException(status_code=404, detail="Propiedad no encontrada")

        if prop["visit_slots"] is None or prop["visit_slots"] <= 0:
            raise HTTPException(status_code=409, detail="Sin cupos disponibles")

 
        request_id = uuidlib.uuid4()
        cur.execute("""
            INSERT INTO purchase_requests (request_id, user_id, group_id, url, origin, operation, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'PENDING')
        """, (str(request_id), user_id, GROUP_ID, data.url, 0, "BUY"))

        cur.execute("UPDATE properties SET visit_slots = visit_slots - 1 WHERE url = %s", (data.url,))


        cur.execute("""
            INSERT INTO event_log (topic, event_type, request_id, url, payload)
            VALUES ('properties/requests', 'REQUEST_SENT', %s, %s, %s::jsonb)
        """, (str(request_id), data.url, json.dumps({
            "request_id": str(request_id),
            "group_id": GROUP_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": data.url,
            "origin": 0,
            "operation": "BUY"
        })))

        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        cur.close(); conn.close()

    # Publicar al broker que sería el RF5
    body = json.dumps({
        "request_id": str(request_id),
        "group_id": GROUP_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": data.url,
        "origin": 0,
        "operation": "BUY"
    })
    ok = mqtt_publish_with_fibonacci(REQUESTS_TOPIC, body)
    if not ok:

        conn = get_connection(); cur = conn.cursor()
        try:
            cur.execute("UPDATE purchase_requests SET status='ERROR', updated_at=CURRENT_TIMESTAMP WHERE request_id=%s", (str(request_id),))
            cur.execute("UPDATE properties SET visit_slots = visit_slots + 1 WHERE url = %s", (data.url,))
            cur.execute("""
                INSERT INTO event_log (topic, event_type, request_id, url, status, payload)
                VALUES ('properties/requests', 'REQUEST_SEND_ERROR', %s, %s, 'ERROR', %s::jsonb)
            """, (str(request_id), data.url, body))
            conn.commit()
        finally:
            cur.close(); conn.close()
        raise HTTPException(status_code=502, detail="No se pudo publicar la solicitud")
    
    # RF01: Generate recommendations when user purchases a visit
    try:
        # Trigger recommendation generation in background
        recommendation_request = RecommendationRequest(
            property_id=str(prop["id"]) if prop else None,
            preferences={
                "price_range": [prop["price"] * 0.8, prop["price"] * 1.2] if prop else None,
                "location": prop["location"].get("address", "") if prop and isinstance(prop["location"], dict) else "",
                "bedrooms": prop["bedrooms"] if prop else None,
                "bathrooms": prop["bathrooms"] if prop else None
            },
            budget_min=prop["price"] * 0.8 if prop else None,
            budget_max=prop["price"] * 1.2 if prop else None,
            location=prop["location"].get("address", "") if prop and isinstance(prop["location"], dict) else None,
            bedrooms=prop["bedrooms"] if prop else None,
            bathrooms=prop["bathrooms"] if prop else None
        )
        
        # Call recommendation generation (this will be processed asynchronously)
        rec_response = requests.post(
            f"{WORKER_SERVICE_URL}/job",
            json={
                "user_id": user_id,
                "property_id": str(prop["id"]) if prop else None,
                "preferences": recommendation_request.preferences,
                "budget_min": recommendation_request.budget_min,
                "budget_max": recommendation_request.budget_max,
                "location": recommendation_request.location,
                "bedrooms": recommendation_request.bedrooms,
                "bathrooms": recommendation_request.bathrooms
            },
            timeout=5
        )
        
        if rec_response.status_code == 200:
            rec_result = rec_response.json()
            # Log recommendation job creation
            conn = get_connection()
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO event_log (topic, event_type, request_id, url, payload)
                    VALUES ('recommendations', 'RECOMMENDATION_JOB_CREATED', %s, %s, %s::jsonb)
                """, (str(request_id), data.url, json.dumps({
                    "recommendation_job_id": rec_result["job_id"],
                    "user_id": user_id,
                    "property_id": str(prop["id"]) if prop else None
                })))
                conn.commit()
            finally:
                cur.close()
                conn.close()
    except Exception as e:
        # Log error but don't fail the visit request
        print(f"Failed to create recommendation job: {str(e)}")
    
    return VisitRequestOut(
        request_id=str(request_id),
        status="PENDING",
        message="Solicitud enviada; tu visita quedó en proceso de validación"
    )

# ===== ENDPOINTS DE COMPRAS (PURCHASES) =====

@app.get("/purchases/{purchase_id}", response_model=PurchaseDetail)
def get_purchase_detail(purchase_id: str, user: dict = Depends(verify_jwt)):
    """
    Devuelve el detalle de una compra específica del usuario autenticado.
    """
    user_id = user.get("sub")
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT pr.request_id, pr.url, pr.status, pr.created_at, pr.amount,
                   pr.status = 'ACCEPTED' AS has_receipt,
                   pr.rejection_reason, pr.authorization_code,
                   p.*
            FROM purchase_requests pr
            LEFT JOIN properties p ON pr.url = p.url
            WHERE pr.request_id = %s AND pr.user_id = %s
        """, (purchase_id, user_id))
        
        r = cur.fetchone()
        
        if not r:
            raise HTTPException(status_code=403, detail="No tienes acceso a esta compra o no existe")
        
        # Construir objeto property con solo los campos de la propiedad
        property_obj = {k: r[k] for k in r.keys() if k not in ["request_id", "url", "status", "created_at", "amount", "has_receipt", "rejection_reason", "authorization_code"]}
        
        return PurchaseDetail(
            request_id=str(r["request_id"]),
            url=r["url"],
            status=r["status"],
            created_at=r["created_at"].isoformat() + "Z",
            amount=float(r["amount"]) if r["amount"] is not None else 0.0,
            has_receipt=bool(r["has_receipt"]),
            property=property_obj,
            rejection_reason=r.get("rejection_reason"),
            authorization_code=r.get("authorization_code")
        )
    finally:
        cur.close()
        conn.close()

@app.get("/purchases/{purchase_id}/receipt")
def get_purchase_receipt(purchase_id: str, user: dict = Depends(verify_jwt)):
    """
    Entrega el PDF de la boleta de compra si la compra está ACCEPTED.
    """
    user_id = user.get("sub")
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT pr.request_id, pr.url, pr.status, pr.created_at, pr.amount,
                   pr.authorization_code, pr.status = 'ACCEPTED' AS has_receipt,
                   p.*
            FROM purchase_requests pr
            LEFT JOIN properties p ON pr.url = p.url
            WHERE pr.request_id = %s AND pr.user_id = %s
        """, (purchase_id, user_id))
        
        r = cur.fetchone()
        
        if not r:
            raise HTTPException(status_code=403, detail="No tienes acceso a esta compra o no existe")
        
        if not r["has_receipt"]:
            raise HTTPException(status_code=403, detail="La compra aún no está aceptada, no hay boleta disponible")
        
        if canvas is None:
            raise HTTPException(status_code=500, detail="reportlab no está instalado")
        
        # Generar PDF
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        pdf.setTitle("Boleta de Compra")
        
        # Título
        pdf.drawString(100, 800, f"Boleta de Compra - ID: {r['request_id']}")
        pdf.drawString(100, 780, f"Propiedad: {r['url']}")
        pdf.drawString(100, 760, f"Monto: ${r['amount']:.2f}")
        pdf.drawString(100, 740, f"Fecha: {r['created_at'].isoformat() + 'Z'}")
        pdf.drawString(100, 720, f"Código de Autorización: {r.get('authorization_code', '-')}")
        pdf.drawString(100, 700, f"Estado: {r['status']}")
        
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=boleta_{r['request_id']}.pdf"}
        )
    finally:
        cur.close()
        conn.close()

# Actualizado: Endpoint para historial con información extendida
@app.get("/my-properties", response_model=list[MyProperty])
def my_properties(user: dict = Depends(verify_jwt)):
    """
    Devuelve las solicitudes de compra del usuario con detalles extendidos.
    """
    user_id = user.get("sub")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    ensure_user_exists(user_id, user.get("name", ""), email, user.get("phone_number", ""))

    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT pr.request_id, pr.url, pr.status, pr.created_at, pr.amount,
                   pr.status = 'ACCEPTED' AS has_receipt,
                   p.*
            FROM purchase_requests pr
            LEFT JOIN properties p ON pr.url = p.url
            WHERE pr.user_id = %s
            ORDER BY pr.created_at DESC
        """, (user_id,))
        
        rows = cur.fetchall()
        result = []
        
        for r in rows:
            # Construir objeto property excluyendo campos de purchase_request
            property_obj = {k: r[k] for k in r.keys() if k not in ["request_id", "url", "status", "created_at", "amount", "has_receipt"]}
            
            result.append(MyProperty(
                request_id=str(r["request_id"]),
                url=r["url"],
                status=r["status"],
                created_at=r["created_at"].isoformat() + "Z",
                amount=float(r["amount"]) if r["amount"] is not None else 0.0,
                has_receipt=bool(r["has_receipt"]),
                property=property_obj
            ))
        
        return result
    finally:
        cur.close()
        conn.close()

# ===== ENDPOINTS DE WEBPAY =====

@app.post("/webpay/create", response_model=WebPayCreateResponse)
def create_webpay_transaction(
    request: WebPayCreateRequest, 
    user: dict = Depends(verify_jwt)
):
    """Crear transacción de WebPay para validar reserva de visita"""
    user_id = user.get("sub")
    name = user.get("name", "")
    NAMESPACE = "https://api.g6.tech/claims"
    email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
    phone = user.get("phone_number", "")
    
    # Asegurar que el usuario existe
    ensure_user_exists(user_id, name, email, phone)
    
    # Validar monto
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0")
    
    # Verificar que la propiedad existe
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT price, visit_slots FROM properties WHERE url = %s", (request.url,))
        prop = cur.fetchone()
        
        if not prop:
            raise HTTPException(status_code=404, detail="Propiedad no encontrada")
        
        if prop["visit_slots"] is None or prop["visit_slots"] <= 0:
            raise HTTPException(status_code=409, detail="Sin cupos disponibles para visita")
        
        # Verificar que el monto es correcto (10% del precio)
        expected_amount = float(prop["price"]) * 0.10
        if abs(request.amount - expected_amount) > 0.01:  # Tolerancia de centavos
            raise HTTPException(
                status_code=400, 
                detail=f"El monto debe ser el 10% del precio de la propiedad: ${expected_amount:.2f}"
            )
        
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()
    
    # Generar IDs únicos
    order_id = f"order_{uuid.uuid4().hex[:12]}"
    session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
    
    # URL de retorno (debe ser configurada según tu frontend)
    return_url = f"{FRONTEND_ORIGIN}/webpay/return?token="
    
    # Crear transacción
    result = webpay_service.create_transaction(
        amount=request.amount,
        order_id=order_id,
        session_id=session_id,
        return_url=return_url
    )
    
    if result["success"]:
        # Solo retornar el token, la transacción se registrará en commit si es exitosa
        return WebPayCreateResponse(
            success=True,
            token=result["token"],
            url=result["url"]
        )
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.post("/webpay/commit", response_model=WebPayCommitResponse)
def commit_webpay_transaction(
    request: WebPayCommitRequest,
    user: dict = Depends(verify_jwt)
):
    """Confirmar transacción de WebPay para reserva de visita"""
    user_id = user.get("sub")
    
    # Confirmar transacción
    result = webpay_service.commit_transaction(request.token)
    
    if result["success"]:
        transaction_data = result["transaction"]
        
        # Verificar si la transacción fue exitosa
        if transaction_data["response_code"] == 0:  # Transacción exitosa
            # Procesar la reserva con la información del request
            conn = get_connection()
            cur = conn.cursor()
            try:
                property_url = request.url
                amount = transaction_data["amount"]
                
                # Verificar que haya cupos
                cur.execute("SELECT visit_slots FROM properties WHERE url = %s", (property_url,))
                prop = cur.fetchone()
                
                if not prop or prop["visit_slots"] <= 0:
                    conn.rollback()
                    raise HTTPException(status_code=409, detail="No hay cupos disponibles")
                
                # Crear purchase_request con información de WebPay
                request_id = uuidlib.uuid4()
                authorization_code = transaction_data.get("authorization_code", "")
                cur.execute("""
                    INSERT INTO purchase_requests (request_id, user_id, group_id, url, origin, operation, status, amount, authorization_code)
                    VALUES (%s, %s, %s, %s, %s, %s, 'PENDING', %s, %s)
                """, (str(request_id), user_id, GROUP_ID, property_url, 0, "BUY", amount, authorization_code))
                
                # Reducir visit_slots
                cur.execute("UPDATE properties SET visit_slots = visit_slots - 1 WHERE url = %s", (property_url,))
                
                # Crear transacción de reserva
                tx_id = f"tx_{uuid.uuid4().hex[:8]}"
                cur.execute("""
                    INSERT INTO transactions (id, user_id, type, amount, description, property_id)
                    VALUES (%s, %s, 'purchase', %s, %s, %s)
                """, (tx_id, user_id, amount, f"Reserva validada vía WebPay: {property_url}", property_url))
                
                # Log del evento
                cur.execute("""
                    INSERT INTO event_log (topic, event_type, request_id, url, payload)
                    VALUES ('properties/requests', 'WEBPAY_VALIDATED_REQUEST_SENT', %s, %s, %s::jsonb)
                """, (str(request_id), property_url, json.dumps({
                    "request_id": str(request_id),
                    "group_id": GROUP_ID,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "url": property_url,
                    "origin": 0,
                    "operation": "BUY",
                    "webpay_validated": True
                })))
                
                conn.commit()
                
                # 🆕 ENVIAR EMAIL DE CONFIRMACIÓN DE PAGO
                user_name = user.get("name", "")
                NAMESPACE = "https://api.g6.tech/claims"
                user_email = user.get(f"{NAMESPACE}/email") or user.get("email", "")
                
                if user_email:
                    try:
                        email_service.send_payment_confirmation(
                            to_email=user_email,
                            user_name=user_name or "Usuario",
                            request_id=str(request_id),
                            property_url=property_url,
                            amount=amount,
                            authorization_code=authorization_code
                        )
                        print(f"📧 Email de confirmación de pago enviado a {user_email}")
                    except Exception as e:
                        print(f"⚠️ Error al enviar email de confirmación: {e}")
                
                # Publicar a MQTT
                body = json.dumps({
                    "request_id": str(request_id),
                    "group_id": GROUP_ID,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "url": property_url,
                    "origin": 0,
                    "operation": "BUY"
                })
                
                ok = mqtt_publish_with_fibonacci(REQUESTS_TOPIC, body)
                
                if not ok:
                    # Revertir cambios si falla MQTT
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute("UPDATE purchase_requests SET status='ERROR', updated_at=CURRENT_TIMESTAMP WHERE request_id=%s", (str(request_id),))
                    cur.execute("UPDATE properties SET visit_slots = visit_slots + 1 WHERE url = %s", (property_url,))
                    cur.execute("""
                        INSERT INTO event_log (topic, event_type, request_id, url, status, payload)
                        VALUES ('properties/requests', 'REQUEST_SEND_ERROR', %s, %s, 'ERROR', %s::jsonb)
                    """, (str(request_id), property_url, body))
                    conn.commit()
                    cur.close()
                    conn.close()
                    raise HTTPException(status_code=502, detail="No se pudo publicar la solicitud")
                
                # CRÍTICO: Publicar validación ACCEPTED porque WebPay ya validó el pago
                validation_body = json.dumps({
                    "request_id": str(request_id),
                    "group_id": GROUP_ID,
                    "seller": 0,
                    "status": "ACCEPTED",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Publicar validación
                validation_ok = mqtt_publish_with_fibonacci(VALIDATION_TOPIC, validation_body)
                
                if not validation_ok:
                    print(f"⚠️ WARNING: No se pudo publicar validación para request_id={request_id}, pero la compra está registrada")
                    # No fallamos aquí porque la compra ya está hecha, solo logueamos el problema
                
                cur.close()
                conn.close()
                
                return WebPayCommitResponse(
                    success=True,
                    request_id=str(request_id),
                    message="Reserva validada y enviada para procesamiento",
                    transaction=transaction_data
                )
                
            except HTTPException:
                conn.rollback()
                raise
            except Exception as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"Error procesando reserva: {str(e)}")
            finally:
                cur.close()
                conn.close()
        else:
            return WebPayCommitResponse(
                success=False,
                error=f"Transacción rechazada. Código: {transaction_data['response_code']}"
            )
    else:
        return WebPayCommitResponse(
            success=False,
            error=result["error"]
        )

@app.get("/webpay/status/{token}")
def get_webpay_status(token: str, user: dict = Depends(verify_jwt)):
    """Obtener estado de transacción WebPay"""
    result = webpay_service.get_transaction_status(token)
    
    if result["success"]:
        return result["transaction"]
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/webpay/return")
def webpay_return(token: str = None):
    """Manejar retorno de WebPay"""
    if not token:
        return {"error": "Token no proporcionado"}
    
    # Obtener estado de la transacción
    result = webpay_service.get_transaction_status(token)
    
    if result["success"]:
        transaction = result["transaction"]
        if transaction["status"] == "AUTHORIZED":
            # Transacción exitosa
            return {
                "success": True,
                "message": "Pago exitoso",
                "transaction": transaction
            }
        else:
            # Transacción fallida
            return {
                "success": False,
                "message": "Pago fallido",
                "transaction": transaction
            }
    else:
        return {
            "success": False,
            "error": result["error"]
        }

# ===== WORKER SERVICE ENDPOINTS =====

@app.post("/recommendations/generate", response_model=RecommendationResponse)
def generate_recommendations(request: RecommendationRequest, user: dict = Depends(verify_jwt)):
    """
    RF01: Generate property recommendations using workers when user purchases a visit
    """
    user_id = user.get("sub")
    
    try:
        # Prepare request for worker service
        worker_request = {
            "user_id": user_id,
            "property_id": request.property_id,
            "preferences": request.preferences or {},
            "budget_min": request.budget_min,
            "budget_max": request.budget_max,
            "location": request.location,
            "bedrooms": request.bedrooms,
            "bathrooms": request.bathrooms
        }
        
        # Call worker service
        response = requests.post(
            f"{WORKER_SERVICE_URL}/job",
            json=worker_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return RecommendationResponse(
                job_id=result["job_id"],
                status=result["status"],
                message=result["message"],
                created_at=result["created_at"]
            )
        else:
            raise HTTPException(
                status_code=502, 
                detail=f"Worker service error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Worker service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@app.get("/recommendations/{job_id}")
def get_recommendation_status(job_id: str, user: dict = Depends(verify_jwt)):
    """
    Get recommendation job status and results
    """
    try:
        response = requests.get(
            f"{WORKER_SERVICE_URL}/job/{job_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(
                status_code=502, 
                detail=f"Worker service error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Worker service unavailable: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get recommendation status: {str(e)}"
        )

@app.get("/worker/heartbeat", response_model=WorkerHeartbeatResponse)
def worker_heartbeat():
    """
    RF04: Check if worker service is available for frontend indicator
    """
    try:
        response = requests.get(
            f"{WORKER_SERVICE_URL}/heartbeat",
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return WorkerHeartbeatResponse(
                status=result["status"],
                timestamp=result["timestamp"],
                service=result["service"],
                workers_active=result["workers_active"]
            )
        else:
            return WorkerHeartbeatResponse(
                status=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                service="JobMaster",
                workers_active=0
            )
            
    except requests.exceptions.RequestException:
        return WorkerHeartbeatResponse(
            status=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            service="JobMaster",
            workers_active=0
        )
    except Exception as e:
        return WorkerHeartbeatResponse(
            status=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            service="JobMaster",
            workers_active=0
        )

