from flask import Flask, render_template, request
import pandas as pd
import joblib
from text_processing.wordopt import wordopt
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = joblib.load("news_classifier_model1.pkl")  # Load the trained model

vectorization = TfidfVectorizer()  # Initialize the TF-IDF vectorizer

def output_label(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "NOT A FAKE NEWS"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        news_text = request.form["news_text"]
        prediction = manual_testing(news_text)
        return render_template("index.html", news_text=news_text, prediction=prediction)
    return render_template("index.html")

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    
    # Use .apply() with a lambda function
    new_def_test["text"] = new_def_test["text"].apply(lambda x: wordopt(x))
    
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RF = model.predict(new_xv_test)
    
    prediction_label = output_label(pred_RF[0])
    return prediction_label

if __name__ == "__main__":
    app.run(debug=True)
