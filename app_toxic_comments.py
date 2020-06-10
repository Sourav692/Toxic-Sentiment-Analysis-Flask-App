# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 04:56:43 2018

@author: dmdm02
"""
################## IMPORTING LIBRARIES AND MODELS ####################
print('Importing libraries...')
import os
import pickle

import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from preprocessing import *

model_dir = './models'

print('Loading model...')
model = load_model("weights_cpu.best.hdf5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = model.input_shape[1]
global graph
graph = tf.get_default_graph()


################## MAKING PREDICTION ####################

# Prediction
def rate_toxic(text):
    text_clean = clean_text(text)
    text_split = text_clean.split(' ')

    # Tokenizer
    sequences = tokenizer.texts_to_sequences(text_split)
    sequences = [[item for sublist in sequences for item in sublist]]

    # Padding
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Prediction
    with graph.as_default():
        predict = model.predict(data).reshape(-1, 1)
    return predict


'''
toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(text)
print('Prediction succesful!')
print(rate_toxic(text))
'''
################### BUILD THE APP ###################
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        txt_input = request.form['comment']
        text_clean = clean_text(txt_input)
        text_split = text_clean.split(' ')
        with open('tokenizer.pickle', 'rb') as handle:
        	tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences(text_split)
        sequences = [[item for sublist in sequences for item in sublist]]
        data = pad_sequences(sequences, maxlen=100, padding='post')
        model = load_model("weights_cpu.best.hdf5")
        predict = model.predict(data).reshape(-1, 1)

		
        #toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(txt_input)
        toxic, severe_toxic, obsence, threat, insult, identity_hate = predict
        '''
        response = {}
        response['Toxic score'] = '%.4f'%toxic
        response['Severe toxic score'] = '%.4f'%severe_toxic
        response['Obsence score'] = '%.4f'%obsence
        response['Threat score'] = '%.4f'%threat
        response['Insult score'] = '%.4f'%insult
        response['Identity hate score'] = '%.4f'%identity_hate
        '''

       # print("toxic")

    return render_template('home.html', Score1='%.4f' % toxic, Score2='%.4f' % threat,
                           Score3='%.4f' % insult, Score4='%.4f' % obsence)


if __name__ == "__main__":
    #   app.run(host='0.0.0.0', port=80)
    app.debug = True
    app.run()
# app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)
