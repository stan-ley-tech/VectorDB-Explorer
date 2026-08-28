import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import benchmarks, collections, search, vectors

app = FastAPI(title="VectorDB Explorer API", version="0.1.0")

allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections.router)
app.include_router(vectors.router)
app.include_router(search.router)
app.include_router(benchmarks.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
