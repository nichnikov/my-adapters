from transformers import list_adapters

adapter_infos = list_adapters(source="ah", model_name='roberta-base')
print(adapter_infos)