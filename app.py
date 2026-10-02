"""Flask web app that labels news article text as FAKE or REAL.

The TF-IDF vectorizer and Passive Aggressive classifier are trained in
`Fake News Detection.ipynb` and loaded here from pickle files.
"""

import os
import pickle

from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safety limit on the size of a single submission (characters).
MAX_INPUT_CHARS = 100_000

# Only load these files if you trust them: unpickling can execute code.
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vector = pickle.load(f)
with open(os.path.join(BASE_DIR, "finalized_model.pkl"), "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if not news:
            return render_template(
                "prediction.html",
                error="Please enter some news article text before clicking Analyze.",
            ), 400

        if len(news) > MAX_INPUT_CHARS:
            return render_template(
                "prediction.html",
                news=news[:MAX_INPUT_CHARS],
                error=f"The text is too long. Please keep it under {MAX_INPUT_CHARS:,} characters.",
            ), 400

        predict = model.predict(vector.transform([news]))[0]

        return render_template(
            "prediction.html",
            news=news,
            prediction_text="Predicted label: {}".format(predict),
        )

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run()
