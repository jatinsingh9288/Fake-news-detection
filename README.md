# Fake News Detection

This is a simple Flask web app that predicts whether a news article is **Fake or Real** using Machine Learning.

## Technologies Used

* Python
* Flask
* Scikit-learn
* HTML/CSS
* Pandas
* Jupyter Notebook

## How It Works

The project uses:

* **TF-IDF Vectorizer** to convert text into numbers
* **Passive Aggressive Classifier** to classify the news

The trained model and vectorizer are saved as `.pkl` files and used by the Flask app.

## Accuracy

The model achieved **95.11% accuracy** on the 20% test split used during evaluation.

## How to Run

Clone the repository:

```bash
git clone https://github.com/jatinsingh9288/Fake-news-detection.git
cd Fake-news-detection
```

Install the requirements:
```bash
pip install -r requirements.txt
```
Run the app:
```bash
python app.py
```
Then open:

```text
http://127.0.0.1:5000
...
## Project Structure
```text
app.py
Fake News Detection.ipynb
finalized_model.pkl
vectorizer.pkl
requirements.txt
templates/
static/
data/
```
## Note
The model is trained on news article text from a dataset. It is only a machine learning prediction and should not be treated as a real fact-checking system.
## Author

**Jatin Singh**
