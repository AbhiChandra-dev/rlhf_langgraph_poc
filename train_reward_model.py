from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset('json', data_files='data/feedback.jsonl')

def preprocess(example):
    example["label"] = 1 if example["feedback"] == "yes" else 0
    return example

dataset = dataset.map(preprocess)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(example):
    return tokenizer(example["query"] + " " + example["response"], truncation=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="models/reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"] if "train" in dataset else dataset["train"][:80],
    eval_dataset=dataset["validation"] if "validation" in dataset else dataset["train"][80:],
)

trainer.train()
