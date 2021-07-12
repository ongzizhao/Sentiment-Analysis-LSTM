# Placeholder. Just add your code here to load data
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences





class Datapipeline():

    def __init__(self):
        #self.train_data_path = train_data_path
        #self.test_data_path = test_data_path
        #tokenizer params
        self.vocab_size = 1000
        self.oov_token = "<OOV>"
        self.max_length = 100
        self.padding_type = "post"
        self.trunction_type="post"


    def fit_tokenizer(self,text):

        """
        return tokenized text 

        Args:
            text (pandas series): pandas series of the text to be tokenized

        return:
            tokenizer: returns a keras tokenizer fit to text input

        """
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        tokenizer.fit_on_texts(text)

        self.tokenizer = tokenizer

        print("saving fitted tokenizer")
        with open('./src/datapipeline/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return tokenizer  # contemplating whether to return an output or just do not return anything
        
    def tokenize_text(self, text, tokenizer ):

        text_sequences = tokenizer.texts_to_sequences(text)

        padded_text_sequences = pad_sequences(
            text_sequences, 
            maxlen = self.max_length,
            padding=self.padding_type, 
            truncating=self.trunction_type
        )

        word_index_text = tokenizer.word_index

        return padded_text_sequences, word_index_text

    def transform_train_data(self,train_data_path):
        """
        return a padded tokenized text and encoded labels, word to index dicitionary of text embeddings

        Args:
            None

        Return:
            padded_text_train (array): return array of token sequences padded to the max length for train set
            ohe_train_y (array): return one hot encoded labels
            word_index_train (dict): the word to index dictionary of the vocabulary for train set
    
        """
        
        train = pd.read_csv(train_data_path)

        #label encode the sentiment col
        train["sentiment"] = train['sentiment'].map({'poor': 0, 'satisfied': 1, 'impressed':2})

        #get train X, train y
        train_X = train.drop(columns='sentiment', axis=1)
        #train_Y = pd.DataFrame(train.loc[:, 'sentiment'])
        train_Y = train[["sentiment"]]

        #make validation and train set

        X_train,X_val,y_train,y_val = train_test_split(train_X,train_Y,test_size=0.15,random_state=24)

        #load embedding
        #with open(self.embedding_data_path,"rb") as handle:
        #    embedding_index = pickle.load(handle)

        #fit the tokenizer
        tokenizer = self.fit_tokenizer(X_train["text"])

        padded_text_train , word_index_train = self.tokenize_text(X_train["text"], tokenizer)
        padded_text_val , _ = self.tokenize_text(X_val["text"], tokenizer)

        #one hot encode labels
        ohe_train_y = tf.keras.utils.to_categorical(y_train, 3)
        ohe_val_y = tf.keras.utils.to_categorical(y_val, 3)

        return padded_text_train, ohe_train_y,padded_text_val,ohe_val_y, word_index_train 

    def transform_test_data(self,test_data_path):
        """
        return a padded tokenized text and encoded labels, word to index dicitionary of text embeddings

        Args:
            None

        Return:
            padded_text_test (array): return array of token sequences padded to the max length for test set
            ohe_test_y (array): return one hot encoded labels
        """

        test = pd.read_csv(test_data_path)

        #label encode the sentiment col
        test["sentiment"] = test['sentiment'].map({'poor': 0, 'satisfied': 1, 'impressed':2})

        #get train X, train y
        test_X = test.drop(columns='sentiment', axis=1)
        test_Y = pd.DataFrame(test.loc[:, 'sentiment'])

        padded_text_test , _ = self.tokenize_text(test_X["text"], self.tokenizer)

        #one hot encode labels
        ohe_test_y = tf.keras.utils.to_categorical(test_Y, 3)

        return padded_text_test, ohe_test_y



