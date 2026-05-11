import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


_MAX_SYSTEM_CHARS = 6000   # ~1,500 tokens
_MAX_HISTORY_CHARS = 8000  # ~2,000 tokens


def call_groq(system_prompt: str, messages: list[dict]) -> str:
    # Truncate system prompt if somehow still too large
    if len(system_prompt) > _MAX_SYSTEM_CHARS:
        system_prompt = system_prompt[:_MAX_SYSTEM_CHARS]

    # Keep only the last 6 messages if history is long (8 turn cap)
    recent_messages = messages[-6:] if len(messages) > 6 else messages

    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in recent_messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        temperature=0.2,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content