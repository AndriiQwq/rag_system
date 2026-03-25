import os
from dotenv import load_dotenv
from groq import Groq
from ..base import Generator
from ...config.settings import settings

load_dotenv()


class GroqGenerator(Generator):
    def __init__(self, model_name: str | None = None, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name or settings.groq_model_name
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content



    