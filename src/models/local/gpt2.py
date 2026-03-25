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
        max_input_length: int | None = None,
    ):
        model_name = getattr(settings, "generation_model_name", "gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens_override = max_new_tokens
        self.do_sample_override = do_sample
        self.temperature_override = temperature
        self.top_p_override = top_p
        self.top_k_override = top_k
        self.no_repeat_ngram_size_override = no_repeat_ngram_size
        self.max_input_length_override = max_input_length
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def _get_param(self, arg, override, setting_name):
        if arg is not None:
            return arg
        if override is not None:
            return override
        return getattr(settings, setting_name, None)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        do_sample: bool = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        no_repeat_ngram_size: int = None,
        max_input_length: int = None,
    ) -> str:
        max_new_tokens = self._get_param(max_new_tokens, self.max_new_tokens_override, "generation_max_new_tokens")
        do_sample = self._get_param(do_sample, self.do_sample_override, "generation_do_sample")
        temperature = self._get_param(temperature, self.temperature_override, "generation_temperature")
        top_p = self._get_param(top_p, self.top_p_override, "generation_top_p")
        top_k = self._get_param(top_k, self.top_k_override, "generation_top_k")
        no_repeat_ngram_size = self._get_param(no_repeat_ngram_size, self.no_repeat_ngram_size_override, "generation_no_repeat_ngram_size")
        max_input_length = self._get_param(max_input_length, self.max_input_length_override, "generation_max_input_length")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length if max_input_length is not None else 512
        ).to(self.device)

        generate_kwargs = dict(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        if do_sample is not None:
            generate_kwargs["do_sample"] = do_sample
        if no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if do_sample:
            if temperature is not None:
                generate_kwargs["temperature"] = temperature
            if top_p is not None:
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
