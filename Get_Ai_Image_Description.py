import os
import io
import base64
from typing import Optional

import streamlit as st
from PIL import Image
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Constants
MAX_RETRIES = 3
WAIT_MULTIPLIER = 1  # seconds

class APIKeyError(Exception):
    """Raised when the GROQ API key is missing."""
    pass

def get_api_key() -> str:
    """Retrieve the GROQ API key from Streamlit secrets or environment variables."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ API Key is missing. Please set it in your environment or Streamlit secrets.")
        raise APIKeyError("Missing GROQ_API_KEY")
    return api_key

def convert_image_to_base64(image_path: str) -> Optional[str]:
    """Convert a local image to a base64-encoded JPEG data URL."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_img}"
    except Exception as e:
        st.error(f"❌ Failed to convert image: {e}")
        return None

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=WAIT_MULTIPLIER))
def get_image_description(image_input: str, is_local: bool = False,
                          model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> Optional[str]:
    """
    Analyze a fashion image and return a descriptive summary.

    Args:
        image_input (str): Path to local image or URL.
        is_local (bool): Whether the input is a local file path.
        model (str): Model name to use for analysis.

    Returns:
        Optional[str]: The descriptive summary, or None on failure.
    """
    prompt = (
    "You are a professional fashion analysis assistant. "
    "Your task is to extract **concise and structured metadata** from fashion product images.\n\n"

    "🧾 Output format (if applicable):\n"
    "- Product Type:\n"
    "- Gender Target:\n"
    "- Color(s):\n"
    "- Fabric or Material:\n"
    "- Style or Cut:\n"
    "- Pattern (if any):\n"
    "- Notable Features (e.g., buttons, zippers, embroidery, logos):\n"
    "- Collar / Neckline Type:\n"
    "- Sleeve Type (if visible):\n"
    "- Fit (e.g., slim, oversized):\n\n"

    "✅ **Rules and Tone**:\n"
    "- Assume the image is clear and detailed.\n"
    "- Describe exactly what you see. Do **not** speculate.\n"
    "- Do **not** use conditional phrases like 'if visible' or 'appears to'.\n"
    "- Use professional fashion terminology.\n"
    "- Be confident, direct, and structured.\n"
     )

    image_url = convert_image_to_base64(image_input) if is_local else image_input
    if not image_url:
        return None

    try:
        client = Groq(api_key=get_api_key())

        with st.spinner("🤔 Analyzing image..."):
            response = client.chat.completions.create(
                model=model,
                temperature=0.3,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )

        if response and response.choices:
            st.success("✅ Image analysis complete!")
            return response.choices[0].message.content

        st.error("❌ Error during get image description, No response received from the model.")
        return None

    except Exception as e:
        st.error(f"❌ Error during get image description: {e}")
        return None
