import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ..base import Generator
from ...config.settings import settings


class GPT2Generator(Generator):
    def __init__(self):
        model_name = settings.gpt2_model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=settings.gpt2_max_input_length
        ).to(self.device)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=min(max_tokens, settings.gpt2_max_new_tokens),
            do_sample=settings.gpt2_do_sample,
            temperature=settings.gpt2_temperature,
            top_p=settings.gpt2_top_p,
            no_repeat_ngram_size=settings.gpt2_no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if settings.gpt2_top_k is not None:
            generate_kwargs["top_k"] = settings.gpt2_top_k
        
        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)
        
        # Decode only the generated part (without the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clear GPU memory cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Simple cleanup
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0].strip()
        
        # If answer is empty, return fallback
        if len(answer) < 5:
            answer = "I don't have enough information to answer this question."
        
        return answer
