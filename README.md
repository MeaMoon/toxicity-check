# Toxic Comment Classification
This project aims on classifying whether the comment submited by the given user is toxic or not by using the trained neural network with ~96% accuracy. This idea is implemented and deployed on Heroku, where anyone can submit their own "data". Which means that this project consists out of three main parts: model creation, flask app and heroku deployment.
## Data
In order to create the model I had to you the data set from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) which contains sentences with possible "hate" related contents and the main idea is to classify whether the comment is: toxic, severely toxic, obscene, threatening, insulting or identity hate related. In orded to help the model understand the data more precisely while training I have used the special embedding file ["glove.6b.300d.txt"](https://www.kaggle.com/thanakomsn/glove6b300dtxt). 
## Data preparation
This part of the project mainly lies inside of the "smaller_toxicity.py" file. There I have loaded the data as well as the embedding file, lowered the text and made all of the values become strings. Then I have created the tokenizer from the train part and saved it to use in the flask app to make user input more understandable for the model. With the help of the tokenizer I changed the text to sequnces and then padded the values to make them all of the same length of 150. The final data preaparation part lies in indexing the words using the embedding file.
## Model Creation
Since the data have been prepared properly I can now create the neural network model. The model type is defined as "Functional" which made further layer concatenation possible. Two main layers are the "Embedding" and "Bidirectional Gated Recurrent Unit(GRU)". Then goes the average and max pooling which are further concatenated into the main Dense layer with "sigmoid" activation function since the task is binary, which means that the "binary_crossentropy" loss function is used. To avoid further possible overfitting is have used the "ModelCheckpoint" and "EarlyStopping" callbacks. ~15000000 training parameters were created using which the model had to train for 1 hour straight and resulted in high performance. Finally, I have saved the weights of this model as "smaller_toxicity.h5". Here is the .png variation of the proposed model structure:
![](./Smaller_NN.png)













