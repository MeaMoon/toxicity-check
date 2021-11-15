# %%
# Used Libraries
import os
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Dense, Input, Embedding, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
# %%
# Loading train and test
train_data = pd.read_csv('data\\train.csv')
test_data = pd.read_csv('data\\test.csv') 
# %%
# Loading the required embedding file
embedding_file = 'embeddd.txt'
# %%
# Specifying all of the 6 toxicity related categories to the y_train
toxicity_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train_data[toxicity_classes].values
# %%
# Lowering the symbols and filling possible "Na" gaps
train_sentences = train_data["comment_text"].fillna("fillna").str.lower()
test_sentences = test_data["comment_text"].fillna("fillna").str.lower()
# %%
# Defining the variables needed for preprocessing
max_features = 100000
max_len = 150
embed_size = 300
# %%
# Creating and saving the tokenizer to use in the web App part
tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(train_sentences))

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
# Changing plain text to sequences using the precreated tokenizer
tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)
# %%
# Padding  the sentences in order to make them suit the required max_len
train_padding = pad_sequences(tokenized_train_sentences, max_len)
test_padding = pad_sequences(tokenized_test_sentences, max_len)
# %%
# Indexing all of the words using embedding_file in order to create a proper Input shape for the NN
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file, errors = 'ignore'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# %%
# Creating the Neural Network
# 1. Defining the input shape
# 2. Embedding the input values
# 3. Main part. Creating the Bidirectional GRU layer
# 4-5. Average and Max pooling layers to avoid further model overfitting
# 6. Cancatenating both of the pooling layers
# 7. Defining the main activation function. Sigmoid is used here, since the task is binary
# 8. Compilimg the model using the binary_crossentropy loss function
# 9. Printing the summary of the whole model
image_input = Input(shape=(max_len, ))
X = Embedding(max_features, embed_size, weights=[embedding_matrix])(image_input)
X = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(X)
avg_pl = GlobalAveragePooling1D()(X)
max_pl = GlobalMaxPooling1D()(X)
conc = concatenate([avg_pl, max_pl])
X = Dense(6, activation="sigmoid")(conc)
model = Model(inputs=image_input, outputs=X)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# %%
# Creating the checkpoint path
# Defining  the EarlyStopping metrics to avoid the overfitting even better
checkpoint_path = r".\toxic_GRU.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]
# %%
# Training the model
# Estimated training time -> 2 hrs
model.fit(train_padding, y_train, batch_size=32, epochs=2, validation_split=0.1, callbacks=callbacks_list)
#4488/4488 [==============================] - 3049s 679ms/step - loss: 0.0359 - accuracy: 0.9066 - val_loss: 0.0446 - val_accuracy: 0.9588
# %%
# Predicting the test part in order to check whether the model is decent
test_values = model.predict([test_padding], batch_size=1024, verbose=1)
# %%
# Plotting the model
# Saving the .h5 version of the created model needed for the App
from keras.utils import plot_model
plot_model(model, to_file='Toxic_NN.png', show_shapes=True, show_layer_names=True)
model.save(".\model\Toxic_NN.h5")
print("Saved model to disk")
# %%
# Not needed part
# Created the .json version of the model 
model_json = model.to_json()
with open(".\model\Toxic_NN.json", "w") as json_file:
    json_file.write(model_json)