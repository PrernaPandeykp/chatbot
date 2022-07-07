# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import speech_recognition as sr

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import tensorflow
lemmatizer = WordNetLemmatizer()
# import pyttsx3 as pp

# #voice
# engine=pp.init()
# voices=engine.getProperty('voices')
# engine.setProperty('voice',voices[1].id)
# def speak(word):
#     engine.say(word)
#     engine.runAndWait()


# chat initialization
model = load_model("chatbot_pretrainedmodel.h5")
intents = json.loads(open("intents.json", encoding='utf-8').read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result



@app.route('/get')
def chatbot_response():
    text = request.args.get('msg')
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    
    return res

if __name__ == "__main__":
    app.run()
