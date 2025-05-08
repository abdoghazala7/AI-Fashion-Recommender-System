from Get_Ai_Image_Description import APIKeyError, get_api_key
from get_vector_recommendetion import recommendations_based_on_vecdb
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any, Union
import streamlit as st
from groq import Groq
import json

class LLMRecommender(APIKeyError):
    def __init__(self):
        """Initialize the LLM recommender."""
        self.api_key = get_api_key()
        self.client = Groq(api_key=self.api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.recommendations_object = recommendations_based_on_vecdb()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1))
    def get_recommendations(self, user_query: str, n: int = 4, k: int = 30) -> Union[str, Dict[str, Any]]:
        """Get LLM-processed recommendations."""
        
        try:
            # Get vector recommendations and user intent
            user_intent, vec_recommendations = self.recommendations_object.get_vector_recommendations(user_query, k=k)

            # Prepare input data
            input_data = {
                "stock_items": [
                    {
                        "id": str(item["id"]),
                        "description": item["content"],
                    }
                    for item in vec_recommendations
                ],
                "user_query": user_intent
            }

            # Create system message
            system_message = f"""
You are a fashion recommendation engine designed to help users discover relevant products based on a natural-language query.

✦ TASK:
From the provided list of stock items (each containing an 'id' and a fashion-related 'description'), select the **top {n} items** that best match the user's intent.

✦ INPUT FORMAT:
You will receive a JSON object with:
- "stock_items": a list of fashion product entries, each with:
    • "id": a unique product identifier
    • "description": a structured string like > solid black jersey top with narrow shoulder straps
- "user_query": a rephrased, natural-language description of what the user is looking for

Example:
{{
  "stock_items": [
    {{
      "id": "4",
      "description": "solid black fitted top in soft stretch jersey with a v-neck and short sleeves"
    }},
    {{
      "id": "15684",
      "description": "solid dark grey long-sleeved jumper in a rib knit with a slightly wider neckline worn details and ribbing around the neckline cuffs and hem"
    }}
    ...
  ],
  "user_query": "I’m looking for a polo shirt that complements my black shoes. Ideally, it should be in a neutral color like gray or navy, perfect for casual outings. I prefer something comfortable and suitable for warm weather."
}}

✦ RULES:
- Select exactly **{n}** items from the "stock_items" list.
- Use only the **'id'** values of the selected items in your output.
- Rank relevance based on alignment with user preferences (e.g., color, style, season, usage).
- Do not fabricate or modify descriptions—choose only from the given input.

✦ OUTPUT FORMAT:
Return a JSON object like this:
{{
  "results": [
    {{ "id": "4" }},
    {{ "id": "1485" }},
    ...
  ]
}}
Ensure this format is followed precisely. No additional text or explanation is allowed outside this JSON object.
"""

            # Create user message
            user_message = f"""
Analyze these fashion items and user query:

{json.dumps(input_data, indent=2, ensure_ascii=False)}

Return recommendations in the exact output format.
"""

            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            results = json.loads(content)

            return user_intent, results

        except Exception as e:
          st.error(f"❌ Error during LLM recommendations: {str(e)}")
