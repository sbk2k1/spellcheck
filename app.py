# setup a flask server on port 3030
from flask import Flask
from textblob import TextBlob 
# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import load_model
# import pickle
import os

app = Flask(__name__)


# def Predict_Next_Words(model, tokenizer, text):

#     sequence = tokenizer.texts_to_sequences([text])
#     sequence = np.array(sequence)
#     preds = np.argmax(model.predict(sequence))
#     predicted_word = ""

#     for key, value in tokenizer.word_index.items():
#         if value == preds:
#             predicted_word = key
#             break

#     print(predicted_word)
#     return predicted_word


# tokenizer = Tokenizer()
# model = load_model('next_words.h5')
# tokenizer = pickle.load(open('token.pkl', 'rb'))


@app.route('/spellcheck/<text>')
def spellcheck(text):
    return str(TextBlob(text).correct())


# @app.route('/suggest/<text>')
# def predict(text):
#     try:
#         text = text.split(" ")
#         text = text[-1:]
#         # print(text)

#         prediction = Predict_Next_Words(model, tokenizer, text)
#         return prediction

#     except Exception as e:
#         # print("Error occurred: ", e)
#         return ""

port= os.environ.get("PORT", 5000)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=port)
