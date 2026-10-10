"""
utils/llm_factory.py
"""
from config.settings import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER, FALLBACK_MODEL


def _groq(model: str, temperature: float, max_tokens: int, reasoning_effort: str | None, max_retries: int):
    from langchain_groq import ChatGroq
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in model else {}
    return ChatGroq(model=model, api_key=GROQ_API_KEY, max_retries=max_retries,
                    temperature=temperature, max_tokens=max_tokens, **extra)


def get_llm(temperature: float = 0.3, max_tokens: int = 4096,
            reasoning_effort: str | None = None, model: str | None = None):
    """
    reasoning_effort ("low"/"medium"/"high") controls how many hidden "thinking"
    tokens a reasoning model spends before answering. Thinking tokens count
    against max_tokens, so "low" leaves more room for the visible answer.
    Only gpt-oss models accept it; for other models it is ignored.

    model overrides GROQ_MODEL (the critic uses a smaller model).
    max_retries lets the Groq client wait and retry when the free tier's
    tokens-per-minute limit is hit, instead of failing the whole trip.

    Fallback: if the main model still fails with a rate limit (e.g. its daily
    allowance is used up), the same call is retried on FALLBACK_MODEL, which
    has its own separate allowance — so the public demo keeps working.
    """
    model = model or GROQ_MODEL
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        print(f"[LLM] Groq → {model}")
        primary = _groq(model, temperature, max_tokens, reasoning_effort, max_retries=5)
        if FALLBACK_MODEL and FALLBACK_MODEL != model:
            from groq import RateLimitError
            backup = _groq(FALLBACK_MODEL, temperature, max_tokens, reasoning_effort, max_retries=3)
            return primary.with_fallbacks([backup], exceptions_to_handle=(RateLimitError,))
        return primary
    from langchain_community.chat_models import ChatOllama
    print("[LLM] Ollama fallback")
    return ChatOllama(model="phi3", temperature=temperature)
