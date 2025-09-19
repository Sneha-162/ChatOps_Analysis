
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import torch

from app.routes.chat_routes import router as chat_router
from app.services.model_service import ModelService

load_dotenv()

# remove global model_service = None (we'll create a local ms during startup)
db_client = None  # keep this so we can close it on shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client

    # --- MongoDB (optional) ---
    mongodb_url = os.getenv("MONGODB_URL")
    if mongodb_url:
        db_client = AsyncIOMotorClient(mongodb_url)
        app.state.database = db_client[os.getenv("DATABASE_NAME", "ChatApp")]
    else:
        # No DB configured; keep the app usable for testing
        app.state.database = None

    # --- Model Service (instantiate it!) ---
    ms = ModelService()
    # If your ModelService defines an async initialize(), uncomment:
    # await ms.initialize()
    app.state.model_service = ms

    try:
        yield
    finally:
        if db_client:
            db_client.close()

app = FastAPI(
    title="ChatOps Analyst API",
    description="AI-powered chat analysis with summarization, sentiment analysis, and insights",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes under /api/v1 (matches your OpenAPI output)
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "ChatOps Analyst API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

# ✅ correct dunder guard
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",                     # explicit import path
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", 8010))
    )