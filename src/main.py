from fastapi import FastAPI
from src.api.routes.qa import router as qa_router

app = FastAPI(title="Vietnamese Legal RAG API", version="0.1.0")
app.include_router(qa_router)

@app.get("/health")
async def health():
    return {"ok": True}