from NER import xlarge
from NER.list_utils import flatten_list
from NER.metrics import calculate_relaxed_metric
from data_utils import load_processed_documents
from seqeval.metrics import *
import json


def specificity(gold, test):
    n = gold.count('O')
    tn = 0
    for g,t in zip(gold, test):
        if g == 'O' and t == 'O':
            tn += 1
    return tn/n

if __name__ == "__main__":

    all_metrics = {}
    for noise in [0.05, 0.1, 0.2, 0.3, 0.6, 1.0]:
        print("==== {} ====".format(noise))

        # file1 = '/Users/gabriel-he/PycharmProjects/LUKE2/synthetic_ner/test_train_data_{}/testa_testb_aggregate_original'.format(noise)

        # file1 = '/Users/gabriel-he/PycharmProjects/luke/data/entity_disambiguation/noise_generated/noise_{}/testa_testb_aggregate_original'.format(noise)

        # file1 =  '/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/LUKE/Pipeline NER + ED/NER/TESTB ONLY/output_{}.csv'.format(noise)
        # file1 = '/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/LUKE/Pipeline NER + ED/NER/TESTB ONLY/output.csv'

        test_documents = load_processed_documents(file1)
        gold_documents = load_processed_documents("testb_only_original")

        test_documents = [document for document in test_documents if "testb" in document["name"]]

        test_labels = [[label[0] for label in document["labels"]] for document in test_documents]
        gold_labels = [[label[0] for label in document["labels"]] for document in gold_documents]

        metrics ={}
        metrics["accuracy"] = accuracy_score(gold_labels, test_labels)
        metrics["precision"] = precision_score(gold_labels, test_labels)
        metrics["recall"] = recall_score(gold_labels, test_labels)
        metrics["f1"] = f1_score(gold_labels, test_labels)

        gold_labels_s = [[label + "-A" if label != 'O' else label for label in l] for l in gold_labels]
        test_predictions_s = [[label + "-A" if label != 'O' else label  for label in l] for l in test_labels]
        metrics["strawberry"] = xlarge.score_from_iob(flatten_list(gold_labels_s), flatten_list(test_predictions_s), False, True)

        metrics['specificity'] = specificity(flatten_list(gold_labels), flatten_list(test_labels))

        # print("==== RELAXED ====")
        relaxed_results = calculate_relaxed_metric(gold_labels, test_labels)

        metrics["precision_relaxed"] = relaxed_results["overall"]["precision"]
        metrics["recall_relaxed"] = relaxed_results["overall"]["recall"]
        metrics["f1_relaxed"] = relaxed_results["overall"]["f1"]

        print(json.dumps(metrics, indent=4))
        print(classification_report(gold_labels, test_labels, digits=4))

        print("\n")
        all_metrics[noise] = metrics

    import pandas as pd

    df = pd.DataFrame.from_dict(all_metrics)
