import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ..base import Generator
from ...config.settings import settings


class GPT2Generator(Generator):
    def __init__(
        self,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        no_repeat_ngram_size: int | None = None,
    ):
        model_name = settings.gpt2_model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens_override = max_new_tokens
        self.do_sample_override = do_sample
        self.temperature_override = temperature
        self.top_p_override = top_p
        self.top_k_override = top_k
        self.no_repeat_ngram_size_override = no_repeat_ngram_size
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        max_new_tokens = self.max_new_tokens_override if self.max_new_tokens_override is not None else settings.gpt2_max_new_tokens
        do_sample = self.do_sample_override if self.do_sample_override is not None else settings.gpt2_do_sample
        temperature = self.temperature_override if self.temperature_override is not None else settings.gpt2_temperature
        top_p = self.top_p_override if self.top_p_override is not None else settings.gpt2_top_p
        top_k = self.top_k_override if self.top_k_override is not None else settings.gpt2_top_k
        no_repeat_ngram_size = (
            self.no_repeat_ngram_size_override
            if self.no_repeat_ngram_size_override is not None
            else settings.gpt2_no_repeat_ngram_size
        )

        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=settings.gpt2_max_input_length
        ).to(self.device)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=min(max_tokens, max_new_tokens),
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
        
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
