import torch
import deepspeed
from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/home/work/llama-30b-hf")
model = AutoModelForCausalLM.from_pretrained("/home/work/llama-30b-hf")
model = deepspeed.init_inference(
    model,
    mp_size=4,
    dtype=torch.half,
    replace_with_kernel_inject=True
)

batch = tokenizer(
    "The primary use of LLaMA is research on large language models, including",
    return_tensors="pt", 
    add_special_tokens=False
)
batch = {k: v.cuda() for k, v in batch.items()}
generated = model.generate(batch["input_ids"], max_length=100)
print(tokenizer.decode(generated[0]))