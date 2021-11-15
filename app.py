# Used libraries
from flask import Flask, request, render_template
from tensorflow import keras
import numpy as np
import os
import pickle
from keras.preprocessing.sequence import pad_sequences


#CURRENT_DIRECTORY = os.dirname(os.abspath(__file__))
path = os.getcwd()
# Defining the Flask app
app = Flask(__name__)
#basepath = os.path.abspath(".")

model = keras.models.load_model(path + "/models/Toxic_NN.h5")

# Loading the saved tokenizer
with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function needed to prepare the input data from the user
def prepare_text(text):
    max_len = 150  
    tokenized_text_sentence = tokenizer.texts_to_sequences(text)
    text_padding = pad_sequences(tokenized_text_sentence, max_len)
    print(text_padding)
    return text_padding

@app.route('/')
def my_form():
    return render_template('alt_hello.html')

# Main part of the project
@app.route('/', methods=['POST'])
def hello():    
    text = request.form['text']        
    prepared_comment = prepare_text(text)    
    print(prepared_comment)    
    result = model.predict([prepared_comment]) 
    percent_result = np.around(result)   
    toxic = percent_result[0,0]
    severe_toxic = percent_result[0,1]
    identity_hate = percent_result[0,2]
    insult = percent_result[0,3]
    obscene = percent_result[0,4]
    threat = percent_result[0,5]          
    return render_template('alt_hello.html', text = text,
                           toxic = toxic,
                           severe_toxic = severe_toxic,
                           identity_hate = identity_hate,
                           insult = insult,
                           obscene = obscene,
                           threat = threat)

if __name__ == '__main__':
    app.run()