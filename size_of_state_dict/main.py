from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForCausalLM

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resnet50_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
print(f"{len(resnet50_model.state_dict())=}")

calm2_model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto")
print(f"{len(calm2_model.state_dict())=}")
