import os
from transformers import (TextClassificationPipeline,
                          RobertaConfig,
                          RobertaModelWithHeads,
                          RobertaTokenizer)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

config = RobertaConfig.from_pretrained("roberta-base", num_labels=2, )
model = RobertaModelWithHeads.from_pretrained("roberta-base", config=config, )
adapter_path = os.path.join(os.getcwd(), "models", "rotten-tomatoes")
model.load_adapter(adapter_path)
model.set_active_adapters("rotten_tomatoes")

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer) #, device=training_args.device.index)
texts = ["complete rubbish",
         "This is awesome!",
         "in general, a good film, but the actors lack screenwriting skills and the director's "
         "imagination so-so"]

for text in texts:
    predict = classifier(text)
    print(predict)
