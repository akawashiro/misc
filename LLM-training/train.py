import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

SEQUENCE_LENGTH = 16
N_EPOCHS = 10
VOCAB_SIZE = 65536

with open("./study-in-scarlet.txt", "r") as f:
    text = f.read()

# Although the vocab size is 50257, we pad it to VOCAB_SIZE to make it a power of 2.
config = LlamaConfig(num_hidden_layers=2, vocab_size=VOCAB_SIZE)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer(text, return_tensors="pt")
inputs_length = inputs["input_ids"].shape[1]
print(f'{inputs["input_ids"].shape=}, {inputs_length=}')

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(N_EPOCHS):
    for i in range(0, inputs_length - SEQUENCE_LENGTH - 1):
        input_ids = inputs["input_ids"][:, i:i+SEQUENCE_LENGTH]
        target_ids = inputs["input_ids"][:, i+1:i+SEQUENCE_LENGTH+1]
        outputs = model(input_ids, labels=target_ids, return_dict=True)
        output_ids = outputs.logits
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        target_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        output_text = tokenizer.decode(output_ids[0].argmax(-1), skip_special_tokens=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'{epoch=} {i=} {loss=}')
        print(f'{input_text=}')
        print(f'{target_text=}')
        print(f'{output_text=}')
