import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ..base import Generator


class GPT2Generator(Generator):
    def __init__(self, model_name: str = "gpt2"):
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
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text.split("Answer:")[-1].strip()
        return answer
