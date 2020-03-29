import os
import json
import pandas as pd


def read_json_data(data_path : str, verbose=True):
    '''reads the contents of a json file, returns the content as a pandas dataframe to facilitate analysis
       json structure: {"data": [{"text": str, "label": str}]}'''

    # later on a check can be implemented to see if the parsed, tokenized and downsampled data is available.
    # but for the purpose of this task with small size, the performance is not affected by redoing it.

    json_path = __get_json_file_path(data_path)

    with open(json_path, encoding='utf8') as file:
        json_data = json.load(file)

    json_data = json_data['data']

    data_frame = pd.DataFrame.from_dict(json_data)

    if verbose:
        print('Parsed json file: ', len(data_frame), 'rows.')

    return data_frame


def store_datasets_as_csv(train : pd.DataFrame, dev : pd.DataFrame, test : pd.DataFrame, path_to_store : str):
    '''stores the provided pandas subsets as csv files'''

    if not os.path.isdir(path_to_store):
        os.makedirs(path_to_store)

    train.to_csv(os.path.join(path_to_store, 'train.csv'), header=False, index=False)
    dev.to_csv(os.path.join(path_to_store, 'dev.csv'), header=False, index=False)
    test.to_csv(os.path.join(path_to_store, 'test.csv'), header=False, index=False)



def __get_json_file_path(data_path):
    '''a helper method to get the name of the json file in the directory'''

    if not os.path.exists(data_path):
        raise ValueError("Data path doesn't exist.")

    json_files = [file for file in os.listdir(data_path) if file.endswith('.json')]

    if not len(json_files):
        raise ValueError('No json files in the provided path.')
    elif len(json_files) > 1:
        raise ValueError('Multiple json files exist. Please keep only one json file in this directory.')

    json_path = os.path.join(data_path, json_files[0])

    return json_path

