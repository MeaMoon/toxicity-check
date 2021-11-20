# Toxic Comment Classification
This project aims on classifying whether the comment submited by the given user is toxic or not by using the trained neural network with ~96% accuracy. This idea is implemented and deployed on Heroku, where anyone can submit their own "data". Which means that this project consists out of three main parts: model creation, flask app and heroku deployment.
## Used Packages
* [numpy](https://numpy.org/install/)
* [os](https://www.geeksforgeeks.org/os-module-python-examples/)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
* [keras](https://pypi.org/project/Keras/)
* [tensorlow](https://www.tensorflow.org/install/pip)
* [pickle](https://wiki.python.org/moin/UsingPickle)
* [flask](https://pypi.org/project/Flask/)
* [requests](https://pypi.org/project/requests/)
* [gunicorn](https://pypi.org/project/gunicorn/)
## Data
In order to create the model I had to use the data set from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) which contains sentences with possible "hate" related contents and the main idea is to classify whether the comment is: toxic, severely toxic, obscene, threatening, insulting or identity hate related. In orded to help the model understand the data more precisely while training I have used the special embedding file ["glove.6b.300d.txt"](https://www.kaggle.com/thanakomsn/glove6b300dtxt). 
## Data preparation
This part of the project mainly lies inside of the "smaller_toxicity.py" file. There I have loaded the data as well as the embedding file, lowered the text and made all of the values become strings. Then I have created the tokenizer from the train part and saved it to use in the flask app to make user input more understandable for the model. With the help of the tokenizer I changed the text to sequnces and then padded the values to make them all of the same length of 150. The final data preaparation part lies in indexing the words using the embedding file.
## Model Creation
Since the data have been prepared properly I can now create the neural network model. The model type is defined as "Functional" which made further layer concatenation possible. Two main layers are the "Embedding" and "Bidirectional Gated Recurrent Unit(GRU)". Then goes the average and max pooling which are further concatenated into the main Dense layer with "sigmoid" activation function since the task is binary, which means that the "binary_crossentropy" loss function is used. To avoid further possible overfitting I have used the "ModelCheckpoint" and "EarlyStopping" callbacks. ~15000000 training parameters were created using which the model had to train for 1 hour straight and resulted in high performance. Finally, I have saved the weights of this model as "smaller_toxicity.h5". 
![Structure of the proposed model](./Smaller_NN.png)
## Flask Web Application
After saving the model I had to create the web application to show its impact. With the help of the Flask package I created a simple backend which was loading the model and saved tokenizer, preparing the text from the user and after using the POST method the submitted and prepared text was evaluated on the model in order to return the output metrics for all of the 6 main toxicity factors.
## Heroku App Deployment
After succesfully running the script on localhost I chose to make this App free to use for anyone. This part took a lot of time, since the created model was too "vast" for the free version, meaning that it was using a lot of RAM. By reducing the amount of training parameters, and specifiying the proper tensorflow package I managed to load this app. Anyone can check it out [here](https://heroku-toxicity-check.herokuapp.com/).
## Used Software
* [Anaconda](https://www.anaconda.com/products/individual)
* Spyder (Python 3.8)
* [Heroku](https://dashboard.heroku.com/apps)












