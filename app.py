import streamlit as st
import joblib
from src.config import MODEL_PATH
from src.predict import predict_posting


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


EXAMPLE_FAKE = {
    "title": "Data Entry Clerk",
    "description": "Urgent hiring! Earn per day from home, no experience needed. Send your phone number and a registration fee to confirm your seat. Contact us on WhatsApp or hr.jobs@gmail.com.",
    "company": "",
    "requirements": "",
}
EXAMPLE_REAL = {
    "title": "Software Developer",
    "description": "We are a software company seeking a developer to join our team and build web applications for clients. You will work with the engineering team on design, testing and deployment of new features.",
    "company": "We build web software for clients across several industries.",
    "requirements": "Two years of experience with JavaScript and databases.",
}


def fill_example(example):
    st.session_state["title_input"] = example["title"]
    st.session_state["description_input"] = example["description"]
    st.session_state["company_input"] = example["company"]
    st.session_state["requirements_input"] = example["requirements"]


st.title("Fake Job Posting Detector")
st.write("Paste a job posting to see whether it looks like a fake one, based on patterns learned from past postings.")

st.sidebar.header("Try an example")
st.sidebar.button("Load fake example", key="example_fake_btn", on_click=fill_example, args=(EXAMPLE_FAKE,), type="primary")
st.sidebar.button("Load real example", key="example_real_btn", on_click=fill_example, args=(EXAMPLE_REAL,), type="primary")

tab_checker, tab_about = st.tabs(["**Checker**", "**About**"])

with tab_checker:
    title = st.text_input("**Job title (optional)**", key="title_input")
    description = st.text_area("**Job description (required)**", height=250, key="description_input")
    company_profile = st.text_area("**Company profile (optional)**", key="company_input")
    requirements = st.text_area("**Requirements (optional)**", key="requirements_input")
    check_clicked = st.button("Check Posting", key="check_button", type="primary")

    if check_clicked:
        if not description.strip():
            st.warning("Please paste a job description first.")
        else:
            with st.spinner("Checking..."):
                pipeline = load_pipeline()
                result = predict_posting(pipeline, title, description, company_profile, requirements)

            if result["label"] == "fake":
                st.error("Likely Fake")
            else:
                st.success("Likely Real")

            percent = result["fake_probability"] * 100
            st.write(f"Fake probability: **{percent:.1f}%**")
            st.progress(result["fake_probability"])

            st.subheader("Red flags")
            if result["red_flags"]:
                for flag in result["red_flags"]:
                    st.write(f"- {flag}")
            else:
                st.write("No obvious red flags found.")

            if result["label"] == "fake" and result["top_words"]:
                st.subheader("Words that pushed the prediction toward fake")
                st.write(", ".join(result["top_words"]))

            st.caption("This is a pattern-based estimate, not a guarantee. Red flags are simple rules, not proof.")

with tab_about:
    st.subheader("About this project")
    st.write(
        "**Dataset:** EMSCAD, 'Real or Fake: Fake Job Posting Prediction' from Kaggle "
        "(about 18,000 postings, about 5% fake)."
    )
    st.write(
        "**Approach:** text is cleaned, converted to TF-IDF numbers, reduced to the 3000 most informative "
        "words, and classified with Logistic Regression tuned by grid search."
    )
    st.write("**Results on the held-out test set (fake class, 173 fake postings out of 3576):**")
    st.write("- Precision: 0.761\n- Recall: 0.902\n- F1: 0.825")
    st.write("**Limitations:**")
    st.write(
        "- The data is old and mostly English, so newer scam styles may be missed.\n"
        "- Red flags are simple rules (heuristics), not proof.\n"
        "- Some words the model learned are specific to this dataset, such as company names.\n"
        "- The same test set was used to compare and select models, so the scores are slightly optimistic.\n"
        "- This is a learning project and not a replacement for your own judgement."
    )
