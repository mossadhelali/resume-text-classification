from data_io import read_json_data, store_datasets_as_csv
from preprocessing import tokenize, downsample, train_dev_test_split
from flair_resume_text_classifier import FlairResumeTextClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import numpy as np
import pandas as pd
import argparse



def main():
    # fix the random seeds, for reproducibility.
    random.seed(1)
    np.random.seed(1)

    # 0. Take path as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    # 1. Read the data
    dataframe_raw = read_json_data(data_path)

    # 2.a pre-analysis
    # view the distribution of classes to determine whether downsampling is needed
    labels_count = dataframe_raw['label'].value_counts().to_dict()
    for i, (label, count) in enumerate(labels_count.items()):
        plt.bar(i+1, count, label=label)
    plt.yticks(list(labels_count.values()))
    plt.ylabel('Num Samples')
    plt.xticks([])
    plt.xlabel('Class')
    plt.title('Distribution of classes in the dataset')
    plt.legend()
    plt.show()

    # as seen from the figure, the class distribution is imbalanced, which might introduce bias towards the none class.
    # one solution is to downsample the none and soft classes to keep the distribution balanced.

    # 2.b pre-processing
    # 2.b.1 Downsampling
    dataframe_downsampled = downsample(dataframe_raw, column_name='label')

    # 2.b.2 Tokenization
    tokenize(dataframe_downsampled)

    # 3. split the data into train/dev/test sets (60/20/20)
    train, dev, test = train_dev_test_split(dataframe_downsampled, column_name='label', dev_test_percentage=0.4)

    # now data is equally sampled, tokenized and split into subsets, time to store it so Flair can start working.
    # 4. save into csvs
    store_datasets_as_csv(train, dev, test, data_path)

    # 5. Initialize FLair corpus, embeddings, model, and trainer
    column_map = {k: v for k, v in enumerate(dataframe_downsampled.columns)}
    model = FlairResumeTextClassifier(data_path, column_map, output_path, 'gpu')

    # 6. train
    model.train()

    # 7. predict the samples of train/dev/test and print accuracy
    train_predictions, dev_predictions, test_predictions = model.get_train_dev_test_predictions(train, dev, test)

    train_acc = accuracy_score(train['label'], train_predictions['prediction'])
    dev_acc = accuracy_score(dev['label'], dev_predictions['prediction'])
    test_acc = accuracy_score(test['label'], test_predictions['prediction'])

    print('Accuracy: Train:', train_acc, 'Dev:', dev_acc, 'Test:', test_acc)

    # 8. Error Analysis
    # Confusion Matrix on Dev
    print('Confusion Matrix:')
    print(pd.DataFrame(confusion_matrix(dev['label'], dev_predictions, labels=['none', 'soft', 'tech']),
                       columns=['none', 'soft', 'tech'], index=['none', 'soft', 'tech']))


if __name__ == '__main__':
    main()