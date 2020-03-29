# Resume Rext Classification

A neural-based solution for the problem of text classification in Resumes.

The goal of this task is to classify short sentences from resumes written in German.
Classes are:
* `tech`: for technical skills such as knowledge in Python, Flair, etc.
* `soft`: for soft skills such as knowledge of English and German
* `none`: for the rest of sentences from resumes such as contact information.


# Approach

### Preprocessing
Because I assumed the `none` class would be dominant, I first do a short pre-analysis to see if it is indeed the case.
It is necessary to do such analysis because imbalance can cause bias towards a specific class. I tackle this problem by downsampling all classes to be equal in size. 
Then, I do a basic word tokenization that doesn't remove any characters because I assume these would be relevant. 
For example, a text that contains @ and + is very likely to belong to the `none` class. 
Finally, I split the downsampled data into train, dev, test (60, 20, 20).  


### Neural Model
For the model, I used the Flair embeddings model. The input is a combination of flair and FastText embeddings, trained
on German Corpora. The combination is fed to a GRU model to classify the text. 
I use the default parameters of the models. There are multiple reasons for the selection of this model:
1. Flair embeddings are contextualized and work even for OOV words.
2. The framework provides pre-trained embeddings that are trained on German corpora.
3. The framework is relatively easy to use and the code is well-maintained.
4. I used Flair embeddings for another task in my thesis, so I am familiar with the API.

The model is able to achieve 93.5% accuracy on the held-out test set, which is in my opinion good enough for such a relatively simple model.

# File Structure 
```
resume-text-classification/
    |__ data/                           a directory where raw json data and intermediate csv files are stored
    |__ data_io.py:                     a module to handle data IO operations
    |__ preprocessing.py:               a preprocessing operations such as downsampling, tokenization and train/dev/test split
    |__ flair_resume_text_classifier:   a wrapper class around Flair objects to handle initialization
    |__ main.py:                        application entry
```    

Note that I would normally not include the data in the repo to protect its privacy, 
but as the provided data is already small in size, I figured it is not worth the trouble.

# Requirements

* Python 3.6
* PyTorch 1.4
* Flair 0.4.5

Other relevant packages such as pandas, scikit and numpy are updated to the latest versions as of 29.03.2020.

# Instructions

To run the analysis from command line:

`python main.py --data-path path/to/data --output-path path/for/output`

`--data-path` is the path to the raw json file and `--output-path` is the directory where the model and its results are saved.

example: 

`python main.py --data-path data/ --output-path output/`

I also provided the same analysis as a Jupyter notebook:

`jupyter notebook resume_text_classification.ipynb`


The model took around 5 minutes to train on a machine with Nvidia GTX TITAN X.

