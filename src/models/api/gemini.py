import os
from dotenv import load_dotenv
import google.generativeai as genai
from ..base import Generator

load_dotenv()


class GeminiGenerator(Generator):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        model = genai.GenerativeModel(
            self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=max_tokens
            )
        )
        response = model.generate_content(prompt)
        return response.text
