import re

def extract_red_flags(text: str) -> list[str]:
    text = text.lower()
    flags = []

    if re.search(r"registration fee|security deposit|processing fee|training fee|pay a fee", text):
        flags.append("Asks for a fee or payment")
    if re.search(r"whatsapp|telegram", text):
        flags.append("Contact only via WhatsApp or Telegram")
    if re.search(r"@(gmail|yahoo|outlook)\.", text):
        flags.append("Uses a free email address")
    if len(text.split()) < 30:
        flags.append("Very short description")
    if re.search(r"earn per day|no experience needed|guaranteed|urgent hiring", text):
        flags.append("Unrealistic or urgent promises")

    return flags


def get_top_words(pipeline, clean: str, n: int = 5) -> list[str]:
    tfidf = pipeline.named_steps['tfidf']
    select = pipeline.named_steps['select']
    clf = pipeline.named_steps['clf']

    vector = select.transform(tfidf.transform([clean]))
    words = tfidf.get_feature_names_out()[select.get_support()]
    contributions = vector.toarray()[0] * clf.coef_[0]

    order = contributions.argsort()[::-1]
    return [str(words[i]) for i in order[:n] if contributions[i] > 0]


from src.config import DECISION_THRESHOLD
from src.preprocess import clean_text


def predict_posting(pipeline, title, description, company_profile, requirements) -> dict:
    raw_text = " ".join([title, company_profile, description, requirements])
    clean = clean_text(raw_text)

    fake_probability = float(pipeline.predict_proba([clean])[0][1])
    label = "fake" if fake_probability >= DECISION_THRESHOLD else "real"

    red_flags = extract_red_flags(raw_text)
    if not company_profile.strip():
        red_flags.append("No company profile provided")

    top_words = get_top_words(pipeline, clean)

    return {
        "label": label,
        "fake_probability": fake_probability,
        "red_flags": red_flags,
        "top_words": top_words,
    }
