# Topic analysis of consumer complaints

This folder is the PyCharm project inside the `Abgabe_2` submission folder. It contains the Python workflow comparing LDA and LSA topic models on Shashwat Tiwari's Kaggle dataset *Consumer Complaints Dataset for NLP*.

The downloaded CSV stays outside this repository in the workspace's `data/raw/` folder, and generated results stay local. The expected full CSV path is `Portfolio-Data-Analysis/data/raw/complaints_processed.csv`, which is `../../data/raw/complaints_processed.csv` relative to this project folder.

## Setup

Use the Project venv selected in PyCharm (Python 3.9–3.12 is supported). Open the PyCharm terminal in this folder and install the pinned packages and English spaCy model:

```sh
python -m pip install -r requirements.txt
```

## Run

From this folder, run:

```sh
python src/analyze_complaints.py --input ../../data/raw/complaints_processed.csv --output results
```

The workflow removes empty narratives, collapses exact duplicate narratives, marks conflicting product labels as ambiguous, tokenizes and lemmatizes the available word sequences, compares CountVectorizer and TF-IDF representations, and fits LDA and LSA models. It evaluates topic counts from 2 through 15 using Gensim's c_v coherence. When scores are within 0.01, it prefers the smaller topic count; top terms and representative narratives are then reviewed for interpretability. It also writes a comparison of vector matrix sizes and sparsity. Results are written to `results/` and are not committed.

The source CSV is already text-preprocessed: it is lowercase and lacks punctuation and common words, including “not”. Original words and sentence boundaries cannot be recovered. Findings therefore describe themes in this processed collection of narratives; they are not estimates of complaints in the population.
