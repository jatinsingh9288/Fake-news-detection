# Fake News Detection (Flask + scikit-learn)

A small web application that classifies **news article text** as **FAKE** or **REAL** using a TF-IDF vectorizer and a Passive Aggressive classifier. The model was trained in a Jupyter notebook and is served through a Flask app.

> This is a learning/portfolio project. Its predictions are statistical guesses from a model trained on one dataset. They are **not** a fact-check.

## Key Features

- Web interface with a Home page and a Prediction page
- Paste news article text into a form and get a `FAKE` or `REAL` label
- Input validation: empty or whitespace-only submissions are rejected with a clear message
- Pre-trained model and vectorizer are loaded from `.pkl` files, so no training and no dataset are needed to run the app
- Full training workflow is available in the notebook (`Fake News Detection.ipynb`)

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Web framework | Flask |
| ML / NLP | scikit-learn (`TfidfVectorizer`, `PassiveAggressiveClassifier`) |
| Data handling (notebook) | pandas, NumPy |
| Frontend | HTML templates (Jinja2), Tailwind CSS via CDN |
| Deployment config | `Procfile` with gunicorn (see note below) |

## How the System Works

1. **Training (notebook):** `news.csv` is loaded, the `text` column (the article body) is used as input and the `label` column (`FAKE`/`REAL`) as the target. Data is split 80/20 with `random_state=20`. A `TfidfVectorizer` is fitted on the training text and a `PassiveAggressiveClassifier` is trained on the result. Both objects are saved with `pickle`.
2. **Serving (`app.py`):** On startup, Flask loads `vectorizer.pkl` and `finalized_model.pkl`.
3. **Prediction:** The form on `/prediction` sends the entered text by POST. The app runs `model.predict(vector.transform([text]))` and shows the returned label on the page.

## Project Structure

```
.
├── app.py                    # Flask application (entry point)
├── Fake News Detection.ipynb # Training and evaluation notebook
├── finalized_model.pkl       # Saved PassiveAggressiveClassifier
├── vectorizer.pkl            # Saved TfidfVectorizer
├── requirements.txt          # Python dependencies
├── Procfile                  # web: gunicorn app:app
├── data/
│   └── README.md             # How to obtain the dataset (not included in the repo)
├── static/
│   └── image.svg
├── templates/
│   ├── index.html            # Home page
│   └── prediction.html       # Prediction form and result
├── .gitignore
└── README.md
```

## Installation and Setup

Requires **Python 3.11 or newer** (scikit-learn 1.8 supports Python 3.11 to 3.14; the notebook metadata shows it was run on Python 3.13.3).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`scikit-learn` is pinned to `1.8.0` because that is the version recorded inside the saved `.pkl` files. Loading them with a different version may show warnings or behave differently.

Only load `.pkl` files from sources you trust: unpickling a file can execute code.

## Running the Flask Application

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

This uses Flask's built-in development server, which is meant for local use only.

Using gunicorn (Linux/macOS), as in the `Procfile`:

```bash
gunicorn app:app
```

## Re-running the Notebook (optional)

The app does not need the dataset, but the notebook does. The dataset is **not included** in this repository; see [`data/README.md`](data/README.md) for what file is expected, where to get it, and how to check you have the same one. Put it in the project root as `news.csv`.

The notebook additionally needs pandas and Jupyter:

```bash
pip install pandas notebook
jupyter notebook "Fake News Detection.ipynb"
```

Running all cells will overwrite `finalized_model.pkl` and `vectorizer.pkl`.

## Model and Preprocessing

| Item | Value (from the notebook / saved files) |
|---|---|
| Vectorizer | `TfidfVectorizer(stop_words='english', max_df=0.7)` |
| Classifier | `PassiveAggressiveClassifier(max_iter=50)` |
| Input column | `text` (full article body) |
| Target column | `label` (`FAKE`, `REAL`) |
| Train/test split | 80% / 20%, `random_state=20` |
| Other preprocessing | None beyond what `TfidfVectorizer` does by default (lowercasing, tokenization, English stop-word removal) |

**Results reported in the notebook (single run, 20% held-out test split of the training dataset, 1,267 articles):**

- Accuracy: **95.11%**
- Confusion matrix (rows = actual, columns = predicted, order FAKE, REAL):

|  | Predicted FAKE | Predicted REAL |
|---|---|---|
| Actual FAKE | 626 | 22 |
| Actual REAL | 40 | 579 |

This figure applies only to that test split: full-length articles from the same dataset the model was trained on. It should **not** be read as the accuracy on other kinds of input (for example short headlines or text from other sources). Precision, recall and F1-score are not computed in the notebook, so they are not reported here.

## Dataset

- File: `news.csv` (not included in this repository, see [`data/README.md`](data/README.md))
- Size: 6,335 rows, 4 columns (`Unnamed: 0`, `title`, `text`, `label`)
- Class balance: 3,171 `REAL` and 3,164 `FAKE`
- Article length: about 776 words per article on average (`text` column)
- No missing values in any column (checked in the notebook)
- Original source / license: not recorded by this project. The file matches the published description of the "Fake or Real News" dataset by George McIntire. Please check the source and license before reuse.

## Example Usage

1. Start the app and open the **Prediction** page.
2. Paste the text of a news article into the input box and click **Analyze**.
3. The page shows the predicted label, for example: `Predicted label: REAL`.

## Screenshots

_Screenshots will be added here._

| Home page | Prediction page |
|---|---|
| `docs/screenshots/home.png` (placeholder) | `docs/screenshots/prediction.png` (placeholder) |

## Limitations

- **Trained on full articles.** The model was trained on the full article body (`text` column, about 776 words on average). Short inputs such as a single headline differ a lot from the training data, so predictions on them are likely to be less reliable than the reported test accuracy. As an informal, one-off check (not part of the notebook), scoring the same saved model on the `title` column of the same 1,267 test rows gave roughly 70% accuracy, far below 95.11%.
- **One dataset, one topic area and time period.** The model learns patterns from a single dataset, so it may not generalise to other sources, topics or time periods. Published descriptions of this dataset say its fake articles come from a 2016 US election-era Kaggle collection and its real articles from mainstream outlets. The model may therefore be partly learning source and style differences rather than truthfulness.
- It does not verify facts or check sources.
- The `Procfile` is included, but no deployment is claimed or documented here.

## Future Improvements

- Evaluate properly on headline-only input, or train a separate headline model
- Report precision, recall and F1-score
- Add basic text cleaning and compare against other classifiers
- Test on articles from other sources and time periods
- Add simple automated tests
- Add a deployment guide once the app is actually deployed

## Author

**Jai Nandan**
GitHub: _Not specified_
LinkedIn: _Not specified_
