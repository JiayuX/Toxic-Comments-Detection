# Toxic-Comments-Detection
In this project, a bidirectional GRU with attention mechanism is used to detect and rate toxic comments.

The training data is from the Toxic Comment Classification Challenge on Kaggle (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

Transfer learning is performed as follows: the pretrained word embeddings GloVe (Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.) is loaded into an embedding layer and is not trained during the training, and the rest of the model is trained with the training data.

All comments are rated in six toxicity categories: toxic, severe_toxic, obscene, threat, insult, identity_hate. The resultant model can be used to predict how toxic a comment is in each toxicity category. 

The model performce could be further improved by experimenting with the achitexture and performing hyperparameter tuning.
