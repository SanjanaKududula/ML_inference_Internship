from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app =Flask(__name__)

###############
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/Review',methods=['get','post'])
def review():
    rev_title=request.form.get("review title")
    rev_text=request.form.get("review text")

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    def clean_text(text):
        sentence = re.sub("[^a-zA-Z]", " ", text)
        text = sentence.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    data_dict = {}
    data_dict["Review Title"]=clean_text(rev_title)
    data_dict["Review text"]=clean_text(rev_text)    


    new_data_df = pd.DataFrame(data_dict, index=[0])

    model = joblib.load("model/decision_tree.pkl")

    sentiment = int(model.predict(new_data_df))

    if sentiment==1:
        message = "Thank you for the Positive review"
    else:
        message = "We're sorry to hear about your disapointment with our product.We'll definitly review your feedback and improve our Product Quality"

    return render_template("home.html", message=message)
###############

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)