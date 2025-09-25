from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional, Union, TypedDict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch, os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

Role = Literal["system", "user", "assistant"]

class ChatMessage(TypedDict):
    role: Role
    content: str
    
Messages = List[ChatMessage]

# Init Configs
@dataclass
class HFLoadConfig:
    model_id: str
    dtype: Literal["auto", "fp16", "bf16"] = "fp16"
    device_map: str="auto"
    load_in_4bit: bool = True
    bnb_4bit_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    max_memory_gib: int = 10
    cpu_offload_dir: str = "offload"
    trust_remote_code: bool = True
    use_sdpa: bool = True

@dataclass
class GenerateConfig:
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = True
    num_beams: int = 1
    stop: Optional[List[str]] = None       # optional stop strings
    max_input_tokens: int = 2048           # truncate input to this
    seed: Optional[int] = None

# Helper
def _to_torch_dtype(name: Literal["auto", "fp16", "bf16"]):
    if name == "auto":
        return "auto"
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return "auto"

# Main Class
class HFModelManager:
    def __init__(self, load_cfg: HFLoadConfig):
        self.cfg = load_cfg
        self.tok = None
        self.model = None
    
    def load(self) -> None:
        if self.tok is not None and self.model is not None:
            return
        quant_cfg = None
        if self.cfg.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_to_torch_dtype(self.cfg.dtype) if self.cfg.dtype != "auto" else torch.float16,
                bnb_4bit_use_double_quant=self.cfg.bnb_4bit_use_double_quant,
                bnb_4bit_type=self.cfg.bnb_4bit_type,
            )
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_id, use_fast=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            # torch_dtype = _to_torch_dtype(self.cfg.dtype),
            dtype= _to_torch_dtype(self.cfg.dtype),
            device_map=self.cfg.device_map,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
            max_memory={0:f"{self.cfg.max_memory_gib}GiB", "cpu": "64GiB"},
            offload_folder=self.cfg.cpu_offload_dir,
            trust_remote_code=self.cfg.trust_remote_code
        )
        if self.cfg.use_sdpa:
            try:
                self.model.config.attn_implementation = "sdpa"
            except Exception:
                pass
            
        self.model.config.use_cache = True
    
    def make_inference(
            self,
            messages_or_prompt: Union[Messages, str],
            gen_cfg: Optional[GenerateConfig] = None,
        ) -> str:
            """
            If `messages_or_prompt` is a string, it's used as-is.
            If it's a list of {role, content}, we apply the model's chat_template.
            """
            self.load()
            assert self.tok is not None and self.model is not None

            gen_cfg = gen_cfg or GenerateConfig()

            if isinstance(messages_or_prompt, str):
                prompt = messages_or_prompt
            else:
                # chat template -> plain text prompt
                prompt = self.tok.apply_chat_template(
                    messages_or_prompt, tokenize=False, add_generation_prompt=True
                )

            # Tokenize with truncation
            inputs = self.tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=gen_cfg.max_input_tokens,
            ).to(self.model.device)

            # Optional seeding for reproducibility
            if gen_cfg.seed is not None:
                torch.manual_seed(gen_cfg.seed)

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    temperature=gen_cfg.temperature,
                    top_p=gen_cfg.top_p,
                    do_sample=gen_cfg.do_sample,
                    num_beams=gen_cfg.num_beams,
                    eos_token_id=self.tok.eos_token_id,
                    use_cache=True,
                )

            # Decode only the newly generated part
            gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
            text = self.tok.decode(gen_tokens, skip_special_tokens=True)

            # Apply optional stop strings
            if gen_cfg.stop:
                text = _truncate_on_stops(text, gen_cfg.stop)

            return text

    def unload(self) -> None:
        """Free large objects (optional)."""
        self.model = None
        self.tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _truncate_on_stops(text: str, stops: List[str]) -> str:
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
            
#     def make_inference(self, messages: List[Dict[str,str]]) -> str:
#         quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_type="nf4",
#                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
#         tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_id, device_map="auto", torch_dtype=torch.float16,
#             quantization_config=quant, low_cpu_mem_usage=True,
#             max_memory={0:"10GiB","cpu":"64GiB"}, offload_folder="offload",
#             trust_remote_code=True
# )
#         model.config.attn_implementation = "sdpa"
#         model.config.use_cache = True
        
#         prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
#         with torch.inference_mode():
#             out = model.generate(
#                 **inputs,
#                 max_new_tokens=256,                
#                 temperature=0.6,
#                 top_p=0.95,
#                 do_sample=True,
#                 num_beams=1,
#                 eos_token_id=tok.eos_token_id,
#                 use_cache=True,
#             )
#         gen = out[0][inputs["input_ids"].shape[-1]:]
#         res = tok.decode(gen, skip_special_tokens=True)
#         return res