# Dataset (not included in this repository)

The Flask app does **not** need the dataset. It only needs `finalized_model.pkl` and
`vectorizer.pkl`, which are in the project root. The dataset is only needed if you want to
re-run `Fake News Detection.ipynb`.

## What the notebook expects

A CSV named `news.csv`, placed in the **project root** (next to the notebook), with:

- 6,335 rows
- columns: `Unnamed: 0`, `title`, `text`, `label`
- `label` values `FAKE` (3,164 rows) and `REAL` (3,171 rows)

`news.csv` is listed in `.gitignore`, so a local copy will not be committed by accident.

## Where it came from

The file used for this project matches the published description of the "Fake or Real News"
dataset by George McIntire (6,335 articles; 3,171 real, 3,164 fake). The author did not
record the exact download source, so verify the source and license yourself before reuse.
Places that host versions of it:

- https://github.com/GeorgeMcIntire/fake_real_news_dataset
- https://github.com/joolsa/fake_real_news_dataset (`fake_or_real_news.csv.zip`)

Different copies of this dataset circulate with different sizes (for example ~10,000 rows
in some repositories). Only a file with the 6,335 rows described above reproduces the
results in the notebook.

## Checking you have the same file

The original `news.csv` used for training has this SHA-256 hash:

```
bb7fa746dd7148b63b0a10e47b329f4b4825afc85b206d3fea18ecfce28ee731
```

Check yours with `sha256sum news.csv` (Linux), `shasum -a 256 news.csv` (macOS) or
`certutil -hashfile news.csv SHA256` (Windows).
