from src.retrieval.pipeline import retrieve_context, build_llm_context, build_sources
from src.services.llm_client import chat_with_local_llm

def _build_user_prompt(question: str, context_text: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context_text}

Yêu cầu:
1) Trả lời ngắn gọn, đúng trọng tâm.
2) Chỉ dùng thông tin trong CONTEXT.
3) Cuối câu trả lời, liệt kê nguồn trích dẫn.
"""

async def ask_question(question: str, top_k: int = 8, debug: bool = False):
    chunks = retrieve_context(
        query=question,
        retrieve_top_k=24,
        final_top_k=top_k,
        bm25_k=30,
        dense_k=30,
        max_chunks_per_doc=2,
    )

    context_text = build_llm_context(chunks)
    sources = build_sources(chunks)
    prompt = _build_user_prompt(question, context_text)
    answer = await chat_with_local_llm(prompt)

    dbg = None
    if debug:
        dbg = {
            "num_chunks": len(chunks),
            "context_chars": len(context_text),
        }

    return {"answer": answer, "sources": sources, "debug": dbg}