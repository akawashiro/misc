from typing import Dict, Union, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def check_past_key_values_tuple(past_key_values_tuple: Tuple) -> None:
    # ((past_key, past_value), ... 32 times)
    assert isinstance(past_key_values_tuple, tuple), f"{type(past_key_values_tuple)=}"
    assert len(past_key_values_tuple) == 32, f"{len(past_key_values_tuple)=}"
    assert isinstance(past_key_values_tuple[0], tuple)
    assert len(past_key_values_tuple[0]) == 2, f"{len(past_key_values_tuple[0])=}"
    assert isinstance(past_key_values_tuple[0][0], torch.Tensor)


def make_past_key_values_tensor(past_key_values_tuple: Tuple) -> torch.Tensor:
    check_past_key_values_tuple(past_key_values_tuple)

    past_key_values_tensors: List[torch.Tensor] = []
    for i in range(32):
        k = past_key_values_tuple[i][0]
        v = past_key_values_tuple[i][1]
        kv = torch.stack([k, v])
        past_key_values_tensors.append(kv)
    past_key_values_tensor = torch.stack(past_key_values_tensors)
    assert isinstance(past_key_values_tensor, torch.Tensor)
    return past_key_values_tensor


def make_past_key_values_tuple(past_key_values_tensor: torch.Tensor) -> Tuple:
    assert past_key_values_tensor.dim() == 6
    past_key_values_tuple = tuple(
        (
            past_key_values_tensor[i][0],
            past_key_values_tensor[i][1],
        )
        for i in range(0, past_key_values_tensor.size(0))
    )
    check_past_key_values_tuple(past_key_values_tuple)
    return past_key_values_tuple


def first_one_step_forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

    onestep_out = model.forward(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True
    )
    logits = onestep_out.logits
    assert isinstance(logits, torch.Tensor)

    past_key_values_tuple = onestep_out.past_key_values
    check_past_key_values_tuple(past_key_values_tuple)
    past_key_values_tensor = make_past_key_values_tensor(past_key_values_tuple)

    ret = {"logits": logits, "past_key_values": past_key_values_tensor}
    return ret


def one_step_forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_key_values_tensor = inputs["past_key_values"]
    past_key_values_tuple = make_past_key_values_tuple(past_key_values_tensor)
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

    onestep_out = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values_tuple,
    )
    logits = onestep_out.logits
    assert isinstance(logits, torch.Tensor)

    past_key_values_tuple = onestep_out.past_key_values
    check_past_key_values_tuple(past_key_values_tuple)
    past_key_values_tensor = make_past_key_values_tensor(past_key_values_tuple)
    past_key_values_tuple = make_past_key_values_tuple(past_key_values_tensor)
    past_key_values_tensor_again = make_past_key_values_tensor(past_key_values_tuple)
    assert torch.allclose(past_key_values_tensor, past_key_values_tensor_again)

    ret = {"logits": logits, "past_key_values": past_key_values_tensor}
    return ret


def main():
    inputs = tokenizer(
        "To be, or not to be, that is the question:", return_tensors="pt"
    )
    GENERATE_LENGTH = 8

    generated_tokens = inputs["input_ids"]
    for i in range(GENERATE_LENGTH):
        print(f'{i=} Decoded input={tokenizer.decode(inputs["input_ids"][0])}')
        if "past_key_values" in inputs:
            print(f"{inputs['past_key_values'].shape=}")

        if i == 0:
            onestep_out = first_one_step_forward(inputs)
        else:
            onestep_out = one_step_forward(inputs)
        logits = onestep_out["logits"]

        next_token = logits.argmax(dim=2)[:, -1:]
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        next_input_ids = next_token
        next_attention_mask = torch.ones_like(next_input_ids)
        next_past_key_values = onestep_out["past_key_values"]

        inputs["input_ids"] = next_input_ids
        inputs["attention_mask"] = next_attention_mask
        inputs["past_key_values"] = next_past_key_values

    print("Generated text:", tokenizer.decode(generated_tokens[0]))


if __name__ == "__main__":
    main()
