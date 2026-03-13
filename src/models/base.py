from abc import ABC, abstractmethod

class Generator(ABC):
    """Base class for all text generators"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text based on prompt"""
        pass
