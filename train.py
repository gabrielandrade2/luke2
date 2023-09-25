import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LukeForEntitySpanClassification, LukeTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments

from data_utils import load_documents, load_examples


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-base")
model.eval()
# model.to("cuda")

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

test_documents = load_documents("eng.testb")
test_examples = load_examples(test_documents, tokenizer)

dataset = CustomDataset([d['words'] for d in test_examples], [d['labels'] for d in test_examples])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

training_args = TrainingArguments(output_dir="test_trainer")
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()