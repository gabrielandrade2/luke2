"""
First, the documents in the dataset is loaded using the `load_documents` function. This function outputs the list of dicts which have the following three keys:
* `words`: the sequence of words
* `labels`: the sequence of gold-standard NER labels (`"MISC"`, `"PER"`, `"ORG"`, or `"LOC"`)
* `sentence_boundaries`: positions of sentence boundaries in the word sequence

The `load_examples` function creates a batch instance for each sentence in a document.
The model addresses the task by classifying all possible entity spans in a sentence into `["NIL", "MISC", "PER", "ORG", "LOC"]`, where `"NIL"` represents that the span is not an entity name (see Section 4.3 in the [original paper](https://arxiv.org/abs/2010.01057)).
Here, we create the list of all possible entity spans (character-based start and entity positions) in a sentence.
Specifically, this function returns the list of dicts with the following four keys:
* `text`: text
* `words`: the sequence of words
* `entity_spans`: the list of possible entity spans (character-based start and end positions in the `text`)
* `original_word_spans`: the list of corresponding spans of `entity_spans` in the word sequence


"""
import unicodedata
from tqdm import tqdm

def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []
    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))

    return documents

def load_processed_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = [0] # Because there is no empty line after DOCSTART
    name = None
    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        name=name,
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = [0]
                name = line[line.find("(")+1:line.find(")")]
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split("\t")
                words.append(items[0])
                if len(items) > 1:
                    labels.append(items[1:])
                else:
                    labels.append(["O"])

    if words:
        documents.append(dict(
            name=name,
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))

    return documents


def load_examples(documents, tokenizer):
    examples = []
    max_token_length = 510
    max_mention_length = 30

    for doc_index, document in tqdm(enumerate(documents)):
        words = document["words"]
        labels = document["labels"]
        if "extra_labels" in document:
            extra_labels = document["extra_labels"]
        else:
            extra_labels = None
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i:i + 2]
            if total_subword_length <= max_token_length:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= max_token_length:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_labels = labels[sentence_start:sentence_end]
            if extra_labels:
                sentence_extra_labels = extra_labels[sentence_start:sentence_end]
            else :
                sentence_extra_labels = None
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                        entity_spans.append(
                            (word_start_char_positions[word_start], word_end_char_positions[word_end])
                        )
                        original_word_spans.append(
                            (word_start, word_end + 1)
                        )

            examples.append(dict(
                text=text,
                words=sentence_words,
                labels=sentence_labels,
                extra_labels=sentence_extra_labels,
                entity_spans=entity_spans,
                original_word_spans=original_word_spans,
                doc_index=doc_index,
            ))

    return examples

def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
