import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("AIによって私達の暮らしは、", return_tensors="pt").to(model.device)  # 入力するプロンプト.
with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,  # 生成する長さ. 128 とかでも良い.
        do_sample=True,
        temperature=0.7,  # 生成のランダム性. 高いほど様々な単語が出てくるが関連性は下がる.
        pad_token_id=tokenizer.pad_token_id,
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)
