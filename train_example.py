import os
import numpy as np
from datasets import load_dataset
from transformers import (RobertaTokenizer,
                          RobertaConfig,
                          RobertaModelWithHeads,
                          TextClassificationPipeline,
                          TrainingArguments,
                          AdapterTrainer,
                          EvalPrediction)


def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


dataset = load_dataset("rotten_tomatoes")
print("num rows:", dataset.num_rows)
print("dataset examples:", dataset['train'][:10], type(dataset))
print("column names:", dataset.column_names)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
# taset.rename_column_("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

config = RobertaConfig.from_pretrained("roberta-base", num_labels=2, )
model = RobertaModelWithHeads.from_pretrained("roberta-base", config=config, )

# Add a new adapter
model.add_adapter("rotten_tomatoes")
# Add a matching classification head
model.add_classification_head("rotten_tomatoes", num_labels=2, id2label={0: "👎", 1: "👍"})
# Activate the adapter
model.train_adapter("rotten_tomatoes")

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

trainer.train()
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=training_args.device.index)
predict = classifier("This is awesome!")
print(predict)
adapter_path = os.path.join(os.getcwd(), "models", "rotten-tomatoes")
model.save_adapter(adapter_path, "rotten_tomatoes")
