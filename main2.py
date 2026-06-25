import base64
import os
import re
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

def _extract_content(response) -> str:
    if isinstance(response, dict):
        return (response.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    if getattr(response, "choices", None):
        return (response.choices[0].message.content or "").strip()
    return ""


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def _exact_n_words(text: str, n_words: int) -> str:
    return " ".join(_words(text)[:n_words]).strip()


def _ensure_sentence_end(text: str) -> str:
    text = text.strip().rstrip(".?!")
    return f"{text}." if text else text


def _client() -> InferenceClient:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY is missing in .env")
    return InferenceClient(api_key=api_key)

def generate_text(prompt: str, max_new_tokens: int = 220) -> str:
    response = _client().chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.4,
    )
    text = _extract_content(response)
    if not text:
        raise RuntimeError("No text returned from the model")
    return text

def generate_exact_sentence(prompt: str, n_words: int, max_new_tokens: int, tries: int = 6) -> str:
    last = ""
    for _ in range(tries):
        last = generate_text(prompt, max_new_tokens=max_new_tokens)
        if len(_words(last)) >= n_words:
            return _ensure_sentence_end(_exact_n_words(last, n_words))
        prompt += f"\n\nTry again. Ensure at least {n_words} words and end with a period."
        time.sleep(0.2)
    return _ensure_sentence_end(_exact_n_words(last, min(n_words, len(_words(last)))))



def main() -> None:
    image_path = input("Image file (default: sample.jpg): ").strip() or "sample.jpg"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = _client()
    try:
        response = client.chat_completion(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Give a short caption for this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ],
            max_tokens=60,
        )
    except Exception as e:
        print("Request failed:", e)
        return

    raw_caption = _extract_content(response)
    if not raw_caption:
        print("Caption: No caption returned")
        return

    caption_prompt = (
        "Rewrite this image description as one clear, natural sentence. "
        "Keep it concise and end with a period.\n\n"
        f"Description: {raw_caption}"
    )
    caption = generate_exact_sentence(caption_prompt, n_words=12, max_new_tokens=80)

    print("Caption:", caption)


if __name__ == "__main__":
    main()