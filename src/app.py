import re
import numpy as np
import tensorflow as tf
from textblob import TextBlob
from flask import Flask,render_template,request
from flask_bootstrap import Bootstrap 
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

MODEL_PATH = "./model/model.h5"
TOKEN_PATH = "./src/datapipeline/tokenizer.pickle"

with open(TOKEN_PATH, 'rb') as handle:
	tokenizer = pickle.load(handle)

app = Flask(__name__)
Bootstrap(app)
model = load_model(MODEL_PATH)

@app.route('/')
def index():
	return render_template('home.html',prediction=float(0.0))


@app.route('/', methods=['GET','POST'])
def predict():
	max_length = 350
	sentiment = None

    # Loading our AI Model
	if request.method == 'POST':
		namequery = request.form['text']
		data = [namequery]
		if data[0] == '':
			prediction = 0
			return render_template('home.html',prediction= float(prediction))
		
		blobline = TextBlob(namequery)
		detect = blobline.detect_language()
		# https://medium.com/analytics-vidhya/flask-language-detector-app-2ee28bfaea4e
		
		if detect == 'en':
			# remove URLs http or https
			result = re.sub ('http\S+','',data[0])

			# remove anything between '<' and '>'
			result = re.sub('<[^>]+>', '', result)
			
			# remove punctuations and convert to lowercase
			#result = re.sub('[^\w\s]','', result).lower()
			
			result = re.sub(r'[\n\r\t]','',result)

			result = re.sub(r'[^\s\w\.]', '', result).lower()

			# remove digits and words containing digits
			result = re.sub('\w*\d\w*','', result)  
			
			# remove spaces from start and end of data
			data_cleaned = re.sub('^\s+', '', result)  
			print ('data after removal is ',data_cleaned)

			tokenizer.fit_on_texts(data_cleaned)
			enc = tokenizer.texts_to_sequences(data)
			enc=pad_sequences(enc, maxlen=max_length, padding='post')
			my_prediction = model.predict(enc) #return list of list of the softmax output eh [["prob1","prob2","prob3"]]
			label = np.argmax(my_prediction)
			label_to_class = {0:"unhappy",1:"satisfied",2:"impressed"}
			sentiment = label_to_class[label]
			prediction = float(my_prediction[0][label])
			#K.clear_session()
		else:
			prediction = "not_en" #not english
	return render_template('home.html',prediction = prediction,sentiment=sentiment, data=data[0],load = "true")
	# K.clear_session()

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/tips')
def tips():
	return render_template('tips.html')

if __name__ == '__main__':
	app.run(debug=True)


