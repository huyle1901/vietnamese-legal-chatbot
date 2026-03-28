import httpx
from src.config import get_settings

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý Việt Nam. "
    "Chỉ trả lời dựa trên CONTEXT được cung cấp. "
    "Nếu không đủ dữ liệu, trả lời rõ là không đủ dữ liệu. "
    "Luôn nêu trích dẫn theo document_number + điều/khoản."
)

async def chat_with_local_llm(user_prompt: str) -> str:
    s = get_settings()
    url = f"{s.local_llm_base_url.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": s.local_llm_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()