'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2023
'''

import csv
import torch

def least_confidence(prob_dist, sorted=False):
    if sorted:
        simple_least_conf = prob_dist.data[0]  # most confident prediction
    else:
        simple_least_conf = torch.max(prob_dist)  # most confident prediction

    num_labels = prob_dist.numel()  # number of labels

    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))

    return normalized_least_conf.item()


def calculate_least_confidence_from_csv(input_csv, output_csv):
    rows = []

    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            img_name, A, B = row
            prob_dist = torch.tensor([float(A), float(B)])
            lc_score = least_confidence(prob_dist)
            rows.append([img_name, A, B, lc_score])

    rows.sort(key=lambda x: x[-1], reverse=True)  # Sort rows by least_confidence, from highest to lowest

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header + ['least_confidence_score'])
        writer.writerows(rows)

input_csv_file = 'unlabeled_probabilities.csv'
output_csv_file = 'least_confidence_scores.csv'

calculate_least_confidence_from_csv(input_csv_file, output_csv_file)