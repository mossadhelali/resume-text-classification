import os
import torch
import flair
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
import pandas as pd


class FlairResumeTextClassifier:
    '''A wrapper around Flair objects that takes care of all initialization'''

    def __init__(self, data_path : str, column_map : {int: str}, output_path : str, device='gpu', verbose=True):
        '''initialize word embeddings, corpus, classifier and model trainer'''

        if device == 'gpu' and torch.cuda.is_available():
            self.embedding_storage = 'gpu'
        else:
            flair.device = torch.device('cpu')
            self.embedding_storage = 'cpu'

        self.verbose = verbose
        self.output_path = output_path

        # 1. initialize corpus
        self.corpus = CSVClassificationCorpus(data_path, column_map,in_memory=True)
        #in_memory=True because of a bug I just discovored in flair that causes a crash. Bug report filed on their repo

        label_dict = self.corpus.make_label_dictionary()

        # 2. initialize embeddings
        # This uses a combination of fasttext word embeddings and flair embeddings for German
        # for the first time only, this will download the embeddings to the local flair caching directory
        if verbose:
            print('Initializing word embeddings from FastText and Flair')

        word_embeddings = [WordEmbeddings('de'), FlairEmbeddings('de-forward'), FlairEmbeddings('de-backward')]
        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                    reproject_words_dimension=256, rnn_type='GRU')

        # 3. initialize classifier and model trainer
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
        self.trainer = ModelTrainer(classifier, self.corpus)


    def train(self):

        num_epochs = 20
        if self.verbose:
            print('Training text classifier for', num_epochs, 'epochs.')

        self.trainer.train(self.output_path, learning_rate=0.1, mini_batch_size=32, anneal_factor=0.5, patience=5,
                           max_epochs=num_epochs, save_final_model=False,
                           embeddings_storage_mode=self.embedding_storage)


    def get_train_dev_test_predictions(self, train, dev, test):
        '''get predictions of train, dev and test sets. Calculate it using the best model
           train, dev and test are dataframes of the original data. Note this has to be provided in order to build
           Sentence objects from scratch without labels.
        '''

        train_sentences = [Sentence(i) for i in train['text']]
        dev_sentences = [Sentence(i) for i in dev['text']]
        test_sentences = [Sentence(i) for i in test['text']]

        classifier = TextClassifier.load(os.path.join(self.output_path, 'best-model.pt'))

        train_sentence_predictions = classifier.predict(train_sentences)
        dev_sentence_predictions = classifier.predict(dev_sentences)
        test_sentence_predictions = classifier.predict(test_sentences)

        train_label_predictions = [sent.labels[0].value for sent in train_sentence_predictions]
        dev_label_predictions = [sent.labels[0].value for sent in dev_sentence_predictions]
        test_label_predictions = [sent.labels[0].value for sent in test_sentence_predictions]

        self.train_predictions = pd.DataFrame(train_label_predictions, columns=['prediction'])
        self.dev_predictions = pd.DataFrame(dev_label_predictions, columns=['prediction'])
        self.test_predictions = pd.DataFrame(test_label_predictions, columns=['prediction'])

        return self.train_predictions, self.dev_predictions, self.test_predictions

