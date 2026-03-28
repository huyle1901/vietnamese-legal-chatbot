from fastapi import APIRouter, HTTPException
from src.api.schemas.qa import AskRequest, AskResponse
from src.services.qa_service import ask_question

router = APIRouter(prefix="/api", tags=["qa"])

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await ask_question(req.question, req.top_k, req.debug)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))