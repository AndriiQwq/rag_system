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
                max_new_tokens=min(max_tokens, 50),  # Shorter answers
                do_sample=True,
                temperature=0.3,  # Very low = more deterministic
                top_p=0.75,  # More restrictive
                top_k=40,  # Limit vocabulary choices
                # repetition_penalty=2.0,  # Strong penalty
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (without the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Simple cleanup
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0].strip()
        
        # If answer is empty, return fallback
        if len(answer) < 5:
            answer = "I don't have enough information to answer this question."
        
        return answer
