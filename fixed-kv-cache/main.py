from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def one_step_forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    onestep_out = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits = onestep_out.logits
    assert isinstance(logits, torch.Tensor)
    ret = {"logits": logits}
    return ret


def main():
    inputs = tokenizer(
        "To be, or not to be, that is the question:", return_tensors="pt"
    )
    GENERATE_LENGTH = 8

    for _ in range(GENERATE_LENGTH):
        print("Decoded input:", tokenizer.decode(inputs["input_ids"][0]))

        onestep_out = one_step_forward(inputs)
        logits = onestep_out["logits"]

        print("Decoded outputs:", tokenizer.decode(logits[0].argmax(dim=1)))

        next_token = logits.argmax(dim=2)[:, -1:]
        next_input_ids = torch.cat([inputs["input_ids"], next_token], dim=1)
        next_attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones((1, 1), dtype=torch.long)], dim=1
        )

        inputs["input_ids"] = next_input_ids
        inputs["attention_mask"] = next_attention_mask


if __name__ == "__main__":
    main()
