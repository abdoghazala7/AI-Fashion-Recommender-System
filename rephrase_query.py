from groq import Groq
import streamlit as st
from textwrap import dedent
from Get_Ai_Image_Description import APIKeyError, get_api_key
from tenacity import retry, stop_after_attempt, wait_exponential

class QueryRephraser:
    def __init__(self, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        """Initialize the query rephraser with model configuration"""
        self.model = model
        self.client = Groq(api_key=get_api_key())
        self.system_prompt = self.get_system_prompt()

    def get_system_prompt(self) -> str:
        """Return a refined system prompt for rephrasing fashion-related user queries."""
        return dedent("""
            You are an expert fashion assistant. Your job is to rephrase and enhance user fashion queries by inferring detailed preferences and adding relevant contextual elements.

            ✦ Tone and Style:
            - Write in a **natural, first-person** voice.
            - Avoid questions or hedging language (e.g., "if applicable").
            - Use confident, fashion-aware phrasing.
            - Keep the output **concise**, **descriptive**, and **visually suggestive**.
            - Infer plausible context when not provided directly.
            - don't ask any Question.          

            ✦ Examples:

            Original: "I want a polo shirt that goes with black shoes."
            Rephrased: "I’m looking for a fitted polo shirt in navy or gray that matches well with my black shoes. It should be casual and versatile, ideal for summer outings."

            Original: "Need a dress for a wedding."
            Rephrased: "I’m looking for an elegant dress for a summer wedding. Soft pastel colors would be perfect, and it should suit a woman who prefers a formal but flattering look."

            Original: "Show me jackets for cold weather."
            Rephrased: "I’m searching for stylish winter jackets that are warm and practical. I need something suitable for a man, ideal for everyday wear in cold weather."

            Use your fashion expertise to fill in missing details, making the query ready for a recommendation engine.
        """)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def rephrase_query(self, query: str) -> str:
        """
        Rephrase and enhance the user's fashion query
        Args:
            query: Original user query
        Returns:
            Enhanced query with structured metadata
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=200,
                top_p=0.9,
                frequency_penalty=0.2
            )
            
            return response.choices[0].message.content

        except APIKeyError as e:
            st.error(f"❌ API Key Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error During Enhanced query: {str(e)}")
