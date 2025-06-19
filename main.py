import os
import logging
import hashlib
import json
import re
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import redis

app = FastAPI()
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KEY = os.getenv("KEY", "")
REDIS_HOST = os.getenv("REDIS_HOST", None)
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

model_cache = {}

rdb = None
if REDIS_HOST:
    try:
        rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        rdb.ping()
        logging.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logging.warning(f"⚠️ Redis unavailable: {e}")
        rdb = None

def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    name = model_name or DEFAULT_MODEL
    if name not in model_cache:
        logging.info(f"Loading model: {name}")
        model_cache[name] = SentenceTransformer(name)
    return model_cache[name]

def build_cache_key(model_name: str, text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    key_base = f"embedding::{model_name}::{normalized}"
    return "cache:" + hashlib.sha1(key_base.encode()).hexdigest()

class TextRequest(BaseModel):
    text: str
    model: Optional[str] = None

class BatchRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None

# 🚀 Warm-up
@app.on_event("startup")
def warmup():
    try:
        model = get_model()
        model.encode("warmup")
        logging.info("✅ Warm-up complete")
    except Exception as e:
        logging.error(f"Warm-up failed: {e}")

@lru_cache(maxsize=10000)
def cached_lru(text: str, model_name: str) -> List[float]:
    model = get_model(model_name)
    return model.encode(text).tolist()

def cached_embedding(text: str, model_name: str) -> List[float]:
    if rdb:
        key = build_cache_key(model_name, text)
        if rdb.exists(key):
            logging.info(f"Redis cache hit: {key}")
            return json.loads(rdb.get(key))
        logging.info(f"Redis cache miss: {key}")
        vec = get_model(model_name).encode(text).tolist()
        rdb.set(key, json.dumps(vec))
        return vec
    else:
        return cached_lru(text, model_name)

def check_api_key(request: Request):
    key = request.headers.get("Key")
    if not KEY or key != KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/embed")
async def embed(req: TextRequest, request: Request):
    check_api_key(request)
    try:
        model_name = req.model or DEFAULT_MODEL
        vec = cached_embedding(req.text, model_name)
        return {
            "embedding": vec,
            "model": model_name,
            "cached": True
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/embed-batch")
async def embed_batch(req: BatchRequest, request: Request):
    check_api_key(request)
    try:
        model = get_model(req.model)
        vectors = model.encode(req.texts).tolist()
        return {
            "embeddings": vectors,
            "model": req.model or DEFAULT_MODEL
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/healthz")
def healthz(model: Optional[str] = Query(None)):
    try:
        _ = get_model(model)
        return {
            "status": "ok",
            "model": model or DEFAULT_MODEL,
            "cache": "redis" if rdb else "lru"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})