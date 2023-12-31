# -*- coding: utf-8 -*-
"""Reproducing LUKE experimental results: CoNLL-2003

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb

# Reproducing experimental results of LUKE on CoNLL-2003 Using Hugging Face Transformers

This notebook shows how to reproduce the state-of-the-art results on the [CoNLL-2003 named entity recognition dataset](https://www.clips.uantwerpen.be/conll2003/ner/) reported in [this paper](https://arxiv.org/abs/2010.01057) using the Trasnsformers library and the [fine-tuned model checkpoint](https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003) available on the Model Hub.
The source code used in the experiments is also available [here](https://github.com/studio-ousia/luke/tree/master/examples/ner).

*Currently, due to the slight difference in preprocessing, the score reproduced in this notebook is slightly lower than the score reported in the original paper (approximately 0.1 F1).*

There are two other related notebooks:

* [Reproducing experimental results of LUKE on Open Entity Using Hugging Face Transformers](https://github.com/studio-ousia/luke/blob/master/notebooks/huggingface_open_entity.ipynb)
* [Reproducing experimental results of LUKE on TACRED Using Hugging Face Transformers](https://github.com/studio-ousia/luke/blob/master/notebooks/huggingface_tacred.ipynb)
"""

# Currently, LUKE is only available on the master branch
import unicodedata

import numpy as np
import seqeval.metrics
import spacy
import torch
from tqdm import tqdm, trange
from transformers import LukeTokenizer, LukeForEntitySpanClassification

from data_utils import load_documents, load_examples

"""## Loading the dataset

The test set of the CoNLL-2003 dataset (eng.testb) is placed in the current directory and loaded using `load_examples` function.
"""

# Download the testb set of the CoNLL-2003 dataset
"""## Loading the fine-tuned model and tokenizer

We construct the model and tokenizer using the [fine-tuned model checkpoint](https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003).
"""

# Load the model checkpoint
model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
model.eval()
# model.to("cuda")

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

## Loading the dataset

test_documents = load_documents("eng.testb")[:5]
test_examples = load_examples(test_documents, tokenizer)

"""## Measuring performance

We classify all possible entity spans in the test set, exclude all spans classified into the `NIL` type, and greedily select a span from the remaining spans based on the logit of its predicted entity type in descending order.
Due to  minor differences in processing, the reproduced performance is slightly lower than the performance reported in the [original paper](https://arxiv.org/abs/2010.01057) (approximately 0.1 F1).
"""

batch_size = 2
all_logits = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    # inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    all_logits.extend(outputs.logits.tolist())

final_labels = [[label for label in document["labels"]] for document in test_documents]

final_predictions = [[] for _ in test_documents]
out_predictions = []
for example_index, example in enumerate(test_examples):
    logits = all_logits[example_index]
    max_logits = np.max(logits, axis=1)
    max_indices = np.argmax(logits, axis=1)
    original_spans = example["original_word_spans"]
    predictions = []
    for logit, index, span in zip(max_logits, max_indices, original_spans):
        if index != 0:  # the span is not NIL
            predictions.append((logit, span, model.config.id2label[index], " ".join(example["words"][span[0]:span[1]]), example["text"], example["doc_index"]))
            out_predictions.append((logit, span, model.config.id2label[index], " ".join(example["words"][span[0]:span[1]]),
                                example["text"], example["doc_index"]))

    # construct an IOB2 label sequence
    predicted_sequence = ["O"] * len(example["words"])
    for _, span, label, _, _, _ in sorted(predictions, key=lambda o: o[0], reverse=True):
        if all([o == "O" for o in predicted_sequence[span[0]: span[1]]]):
            predicted_sequence[span[0]] = "B-" + label
            if span[1] - span[0] > 1:
                predicted_sequence[span[0] + 1: span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

    final_predictions[example['doc_index']] += predicted_sequence

print(seqeval.metrics.classification_report(final_labels, final_predictions, digits=4))

with open("output.csv", "w") as out_file:
    for p in out_predictions:
        out_file.write("\t".join([str(p[5]), str(p[5]), p[3], p[4]]))
        out_file.write("\n")