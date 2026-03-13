def get_generator(generator_type: str):
    """Factory function to create the appropriate generator."""
    if generator_type == "gpt2":
        from .local.gpt2 import GPT2Generator
        return GPT2Generator()
    if generator_type == "gemini":
        from .api.gemini import GeminiGenerator
        return GeminiGenerator()
    if generator_type == "groq":
        from .api.groq import GroqGenerator
        return GroqGenerator()
    raise ValueError(f"Unknown generator type: {generator_type}")
