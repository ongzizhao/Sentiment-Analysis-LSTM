import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

from src.datapipeline.loader import Datapipeline

class Model():

    def __init__(self,embedding_data_path, word_index):
        self.embedding_data_path = embedding_data_path
        self.max_length = 100
        self.word_index = word_index

    def get_embedding_matrix(self):

        with open(self.embedding_data_path, 'rb') as handle:
            embeddings_index = pickle.load(handle)

        embedding_matrix = np.zeros((len(self.word_index) + 1, self.max_length)) 

        # for each word in out tokenizer lets try to find that work in our w2v model
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
        return embedding_matrix
    
    def load_model(self,model_data_path):

        self.trained_model = keras.models.load_model(model_data_path,custom_objects=None)

        return self.trained_model

    def get_model(self):

        embedding_matrix = self.get_embedding_matrix()

        embedding_layer = Embedding(len(self.word_index) + 1,
                            self.max_length,
                            weights=[embedding_matrix],
                            input_length=self.max_length,
                            trainable=False)

        """
        embedding glove - lstm
        https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        """

        DROPOUT_RATE = 0.2

        model = Sequential()
        model.add(embedding_layer)

        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_RATE))

        model.add(Bidirectional(LSTM(64)))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_RATE))

        # model.add(Bidirectional(LSTM(16, return_sequences=True)))
        # model.add(Dropout(DROPOUT_RATE))
        # model.add(BatchNormalization())
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(DROPOUT_RATE))

        model.add(Dense(3, activation='softmax'))

        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self,X_train,y_train,X_val,y_val):
        
        model = self.get_model()

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1, restore_best_weights=True), #patience 10 epochs
            tf.keras.callbacks.ModelCheckpoint(filepath= './Model_checkpoint/model.h5',
                                            monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, verbose=1) #factor learning rate is reduced 
        ]
        
        history = model.fit(
            X_train, 
            y_train,         
            validation_data = (X_val, y_val),
            epochs=20,
            callbacks=my_callbacks,
            verbose=1)
        
        overall_accuracy = history.history["accuracy"][-1]
        
        
        # Add your implementation
        return overall_accuracy


def predict_text(text, tokenizer_path, model_path):
    """
    Tokenize the text with the tokenizer ( pickle file)

    Args:
        text (list of string): input text must be a list of strings 
        tokenizer_path (string): data path to the tokenizer pickle file

    """

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    
    #load Datapipeline class to tokenize input
    dpipe = Datapipeline()
    padded_text, _ = dpipe.tokenize_text(text,tokenizer)

    #load model class to get pretrained model
    pretrained_model = keras.models.load_model(model_path,custom_objects=None)

    #Return prediciton
    pred = pretrained_model.predict(padded_text)
    label = np.argmax(pred)

    label_to_sentiment = {0:"poor",1:"satisfied",2:"impressed"}

    sentiment = label_to_sentiment[label]


    return sentiment


