# Toxic-Comments-Detection
In this project, a toxic comments detector is built and tested.

The training data is from the Toxic Comment Classification Challenge on [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Machine Learning Platforms
PyTorch and Tensorflow/Keras are used independently to provide two versions of solution to this project. 

Text data preprocessing is done using a customized class and functions written in the 'preprocessing.py' file. Tensorflow and Keras are used to build and train the model.

## Model details
A bidirectional GRU with attention mechanism is used to build the model. Transfer learning is performed as follows: the pretrained word embeddings GloVe (Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.) is loaded into an embedding layer and is not trained during the training, and the rest of the model is trained with the training data.

All comments are rated in six toxicity categories: toxic, severe_toxic, obscene, threat, insult, identity_hate. The resultant model can be used to predict how toxic a comment is in each toxicity category. 

The model performance could possibly be further improved by experimenting with the achitecture, conducting hyperparameter tuning and using word embeddings with higher dimensions.
