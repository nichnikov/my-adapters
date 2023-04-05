import os
from transformers import (AutoTokenizer,
                          AutoModel,
                          RobertaConfig,
                          RobertaModelWithHeads,
                          TextClassificationPipeline,
                          AutoModelWithHeads)


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
config = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', num_labels=2, )
model = AutoModelWithHeads.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                                                        config=config, )


adapter_path = os.path.join(os.getcwd(), "models", "rotten-tomatoes2")
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
