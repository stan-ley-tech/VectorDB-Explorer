from fastapi import FastAPI

app = FastAPI(title="VectorDB Explorer API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
