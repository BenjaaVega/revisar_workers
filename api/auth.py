import os
from fastapi import Request, HTTPException, status, Depends
from jose import jwt, JWTError
from typing import Dict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests

# Configuración de Auth0
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "dev-n5t4wuedvu54i50n.us.auth0.com")
API_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://api.g6-arquisis.com")
ALGORITHMS = ["RS256"]

# Obtener y cachear las llaves públicas de Auth0
jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# Cache de JWKS para evitar requests repetidos
jwks_cache = None
jwks_cache_time = None
JWKS_CACHE_DURATION = 3600  # 1 hora en segundos

def get_jwks():
    """Obtener JWKS con cache para evitar requests repetidos"""
    global jwks_cache, jwks_cache_time
    import time
    
    current_time = time.time()
    
    # Si no hay cache o el cache expiró, obtener nuevas llaves
    if jwks_cache is None or (jwks_cache_time and current_time - jwks_cache_time > JWKS_CACHE_DURATION):
        try:
            response = requests.get(jwks_url, timeout=10)  # Timeout de 10 segundos
            response.raise_for_status()
            jwks_cache = response.json()
            jwks_cache_time = current_time
        except requests.RequestException as e:
            # Si falla, usar cache anterior si existe
            if jwks_cache is None:
                raise HTTPException(status_code=503, detail=f"Error obteniendo llaves de Auth0: {str(e)}")
    
    return jwks_cache

security = HTTPBearer()

def get_public_key(token: str):
    from jose.utils import base64url_decode
    unverified_header = jwt.get_unverified_header(token)
    jwks = get_jwks()
    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            return jwt.construct_rsa_public_key(key)
    raise HTTPException(status_code=401, detail="No se encontró la llave pública adecuada.")

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    token = credentials.credentials
    try:
        # Obtener JWKS con cache
        jwks = get_jwks()
        
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        if rsa_key:
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=ALGORITHMS,
                audience=API_AUDIENCE,
                issuer=f"https://{AUTH0_DOMAIN}/"
            )
            return payload
        else:
            raise HTTPException(status_code=401, detail="No se encontró la llave pública adecuada.")
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
