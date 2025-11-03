# api/Dockerfile
FROM python:3.11-slim
# Evitar problemas de compilación con psycopg2
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /app

# Copiar requirements y luego instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Comando por defecto para correr FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
