import base64
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"


def main() -> None:
    image_path = input("Image file (default: sample.jpg): ").strip() or "sample.jpg"
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("HF_API_KEY is missing in .env")
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = InferenceClient(api_key=api_key)
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

    caption = ""
    if isinstance(response, dict):
        caption = (response.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    elif getattr(response, "choices", None):
        caption = (response.choices[0].message.content or "").strip()

    print("Caption:", caption or "No caption returned")


if __name__ == "__main__":
    main()