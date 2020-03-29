import pandas as pd
from segtok.tokenizer import word_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def tokenize(dataframe : pd.DataFrame):
    '''tokenize each row text using the simple sigtok word tokenizer'''

    for _, row in dataframe.iterrows():
        tokenized_text = ' '.join(word_tokenizer(row['text']))
        row['text'] = tokenized_text


def downsample(dataframe : pd.DataFrame, column_name : str):
    '''in case of class imbalance, downsample the dominant classes to remove the bias'''

    label_counts = dataframe[column_name].value_counts().to_dict()

    min_num_samples = min(label_counts.values())

    downsampled_subsets = []
    for label in label_counts.keys():
        label_downsampled = resample(dataframe[dataframe[column_name] == label], replace=False, n_samples=min_num_samples,
                                     random_state=1)
        downsampled_subsets.append(label_downsampled)

    dataframe_downsampled = pd.concat(downsampled_subsets)
    return dataframe_downsampled


def train_dev_test_split(dataframe : pd.DataFrame, column_name : str, dev_test_percentage=0.4):
    '''split a dataframe into train, dev, test sets and stratify based on column_name'''
    train, dev_test = train_test_split(dataframe, shuffle=True, test_size=dev_test_percentage, stratify=dataframe[column_name], random_state=1)
    dev, test = train_test_split(dev_test, test_size=0.5, stratify=dev_test[column_name], random_state=1)

    return train, dev, test