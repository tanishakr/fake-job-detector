# Fake Job Posting Detector

A web app that checks a pasted job posting and estimates whether it is likely fake or likely real.

**Live app:** https://fake-job-detector-26.streamlit.app/

Free Streamlit apps sleep when nobody is using them, so the first load may take a moment while it wakes up.

## Problem

Fake job postings try to collect money or personal details from job seekers. This project trains a classical machine learning model on past postings to pick up the language patterns that fake ones tend to use.

## What the app does

Paste a job posting (only the description is required) and the app shows:

- a verdict: **Likely Fake** or **Likely Real**
- the fake probability as a percentage
- rule-based red flags, such as asking for a fee, contact only via WhatsApp or Telegram, a free email address, a missing company profile, a very short description, or unrealistic promises
- the words that pushed the prediction toward fake (shown for fake predictions only)

The red flags are simple rules, not proof. The result is a pattern-based estimate, not a guarantee.

## Dataset

"Real / Fake Job Posting Prediction" (EMSCAD) by Shivam Bansal on Kaggle, licensed CC0 (public domain). The data originates from the University of the Aegean, Laboratory of Information & Communication Systems Security.

- 17,880 postings, of which 866 (about 4.8%) are fake, so the classes are heavily imbalanced
- Text columns used: title, company profile, description, requirements, benefits
- Split: 80% train, 20% test, stratified so both sets keep the same fake fraction (test set: 3,576 postings, 173 fake)
- The CSV is not included in this repository

## Approach

1. **Cleaning:** join the text columns, lowercase, strip HTML tags, replace symbols with spaces.
2. **TF-IDF:** turn the text into numbers. The final model uses single words.
3. **Feature selection:** SelectKBest (chi-squared) keeps the 3,000 most informative words.
4. **Model:** Logistic Regression with `class_weight='balanced'` to handle the imbalance.
5. **Tuning:** GridSearchCV with stratified 5-fold cross-validation, scored on F1 for the fake class.

The primary metric is F1 for the fake class. Accuracy is not used to choose a model, because predicting "real" for everything would already score about 95%.

Final tuned settings: `max_features=10000`, `ngram_range=(1,1)`, `k=3000`, `C=10`.

## Results

Scores for the fake class on the test set:

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Baseline (TF-IDF + Logistic Regression, no selection) | 0.739 | 0.902 | 0.812 |
| With feature selection (k=3000) | 0.617 | 0.913 | 0.737 |
| Linear SVM with feature selection (k=3000) | 0.725 | 0.884 | 0.797 |
| **Final: tuned Logistic Regression + selection** | 0.761 | 0.902 | 0.825 |

Confusion matrix of the final model: 3354 real postings correctly kept, 49 real flagged as fake, 17 fake missed, 156 fake caught.

How to read these numbers honestly:

- Feature selection at k=3000 lowered F1 compared with the baseline (0.737 vs 0.812).
- The tuned model is only slightly better than the baseline (0.825 vs 0.812). With 173 fake postings in one test set, this small gap may not be meaningful.
- The best values found for `C` (10) and `k` (3000) were the largest in the grid, so better settings may exist beyond it. The grid always kept the selection step, so "no selection" was not tuned.
- Models were compared and selected using the same test set, so the final score is slightly optimistic.

## Run locally

```bash
git clone https://github.com/tanishakr/fake-job-detector.git
cd fake-job-detector
pip install -r requirements.txt
python -m streamlit run app.py
```

The model was trained with scikit-learn 1.6.1, which is pinned in `requirements.txt`. The app was tested on Python 3.12.

## Project structure

```
app.py                  Streamlit app
src/                    config, text cleaning, prediction code
model/                  saved trained pipeline
notebooks/              EDA, preprocessing, modeling
.streamlit/config.toml  app colour theme
requirements.txt
```

## Limitations

- The data is old and mostly English, so the model may not recognise newer scam styles or other languages.
- Some words the model learned look dataset-specific (for example company names), not general scam language.
- The red flags are simple rules and can miss scams or flag normal postings.
- The app does not use the benefits text, company logo or screening-question fields.
- This is a learning project. It is not a replacement for your own judgement when checking a job offer.
