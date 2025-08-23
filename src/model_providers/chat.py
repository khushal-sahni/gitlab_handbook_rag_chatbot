from typing import List
from ..config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from GitLab's Handbook and Direction pages. "
    "If the answer is not found in the provided context, say you don't know and suggest where to look. "
    "Always cite sources with their URLs."
)

def get_chat_fn():
    if PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        def _chat(messages: List[dict], temperature=0.2):
            # messages: [{"role":"system/user/assistant", "content": "..."}, ...]
            # Gemini SDK uses a single prompt; concatenate with roles.
            sys = SYSTEM_PROMPT
            convo = []
            if messages and messages[0].get("role") == "system":
                sys = messages[0]["content"] + "\n" + SYSTEM_PROMPT
                messages = messages[1:]
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                convo.append(f"{role.upper()}: {content}")
            prompt = sys + "\n\n" + "\n".join(convo)
            resp = model.generate_content(prompt, generation_config={"temperature": temperature})
            return resp.text
        return _chat
    elif PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        def _chat(messages: List[dict], temperature=0.2):
            out = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                temperature=temperature,
            )
            return out.choices[0].message.content
        return _chat
    else:
        raise RuntimeError(f"Unsupported PROVIDER: {PROVIDER}")
