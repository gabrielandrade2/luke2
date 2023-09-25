import json
import os

from tqdm import tqdm
from NER import noise
from src.model.efficient_el import EfficientEL

if __name__ == '__main__':

    noise_ratio = 0.05

    model = EfficientEL.load_from_checkpoint("models/pre_trained_model.ckpt").eval()
    model.hparams.threshold = -3.2
    model.hparams.test_with_beam_search = False
    model.hparams.test_with_beam_search_no_candidates = False

    tokenizer = model.tokenizer
    for i in tqdm(range(10)):
        with open('data/aida_val_dataset.jsonl', 'r') as jsonl_file:
            jsonl_content = jsonl_file.readlines()
            file = [json.loads(jline) for jline in jsonl_content]


        for entry in file:

            batch = {
                f"src_{k}": v.to('cpu')
                for k, v in tokenizer(
                    entry['input'],
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding=True,
                    max_length=model.hparams.max_length,
                    truncation=True,
                ).items()
            }

            offsets = batch["src_offset_mapping"]

            new_start = [offsets[0][anchor[0]][0].item() for anchor in entry['anchors']]
            new_end = [offsets[0][anchor[1]][1].item() for anchor in entry['anchors']]
            start_char, end_char = noise.add_noise_larger_single_word_probability(entry['input'], new_start, new_end, noise_ratio)
            new_start = [(offsets[0, :, 0] == index).nonzero()[0].item() for index in start_char]
            new_end = [(offsets[0, :, 1] == index).nonzero()[-1].item() for index in end_char]

            for anchor, s, e, sc, ec in zip(entry['anchors'], new_start, new_end, start_char, end_char):
                anchor[0] = s
                anchor[1] = e

        os.makedirs('data/noise_val_{}'.format(noise_ratio), exist_ok=True)

        with open('data/noise_val_{}/aida_val_dataset_noise_{}.jsonl'.format(noise_ratio, i), 'w', encoding='utf8') as fout:
            for entry in file:
                json.dump(entry, fout, ensure_ascii=False)
                fout.write('\n')
