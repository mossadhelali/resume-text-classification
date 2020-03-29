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
Since this problem is indeed present, I tackle it by downsampling all classes to be equal in size. 
Then, I do a basic word tokenization that doesn't remove any characters because I assume these would be relevant. 
For example, a text that contains @ and + is very likely to belong to the `none` class. 
Finally, I split the downsampled data into train, dev, test (60, 20, 20).  


### Neural Model
For the model, I used the Flair embeddings model. The input is a combination of flair and FastText embeddings, trained
on German Corpora. The combination is fed to a GRU model to classify the text. 
I use the default parameters of the models. There are multiple reasons for the selection of this model:
1. Flair embeddings are contextualized and work even for OOV words.
2. The framework provides pre-trained embeddings that are run on German corpora.
3. The framework is relatively easy to use and the code is well-maintained.
4. I used Flair embeddings for another task in my thesis, so I am familiar with the API.

The model is able to achieve 92.9% accuracy on the held-out test set.

# File Structure 
```
resume-text-classification/
    |__ data/                   a directory where data for each langauge is kept. Each language is stored as a sperate txt file with the langauge name.
    |__ data_reader.py:         a module to handle data reading and formatting
    |__ feature_extractor.py:   a module to handle feature extraction  
    |__ main.py:                implementation of model fitting and analysis
```    

Note that I would normally not include the data in the repo to protect its privacy, 
but as the provided data is already small in size, I figured it is not worth the trouble.

# Requirements

* Python 3.6
* PyTorch 1.4
* Flair 0.4.5


# Instructions

To run the analysis from command line:

`python main.py --data-path path/to/data --output-path path/for/output`

example: 

`python main.py --data-path data/ --output-path output/`

I also provided the same analysis as a Jupyter notebook:

`jupyter notebook resume_text_classification.ipynb`


For me, the model took around 5 minutes to finish on a machine with Nvidia GTX TITAN X.

