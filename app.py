import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="⚓", layout="centered")

# ---------- Look and feel: light, clean, ocean-blue cards ----------
st.markdown(
    """
    <style>
    .stApp { background: #eef4f8; }
    .stApp p, .stApp label, .stApp span, .stApp li { color: #16324f; }
    .hero {
        background: linear-gradient(100deg, #0b3c5d 0%, #1d7a8c 100%);
        border-radius: 16px; padding: 26px 30px; margin-bottom: 22px;
    }
    .hero h1 { color: #ffffff !important; margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p  { color: #cfe8ee !important; margin: 6px 0 0 0; font-size: 1rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border: 1px solid #d5e3ec; border-radius: 14px;
    }
    .q { color: #0b3c5d; font-weight: 700; font-size: 0.95rem; margin-bottom: 2px; }
    .q b {
        display: inline-block; background: #1d7a8c; color: #fff; width: 24px; height: 24px;
        line-height: 24px; text-align: center; border-radius: 50%; margin-right: 8px; font-size: 0.8rem;
    }
    .summary { background: #ffffff; border: 1px solid #d5e3ec; border-radius: 14px; padding: 18px 20px; }
    .summary h4 { color: #0b3c5d; margin: 0 0 10px 0; }
    .row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #e6eef4; color: #16324f; }
    .row:last-child { border-bottom: none; }
    .row span:last-child { font-weight: 700; color: #1d7a8c; }
    .stButton > button {
        width: 100%; background: #0b3c5d; color: #ffffff; border: none; border-radius: 10px;
        padding: 0.7rem 1rem; font-weight: 600; font-size: 1rem; margin-top: 10px;
    }
    .stButton > button:hover { background: #1d7a8c; color: #ffffff; }
    .verdict { border-radius: 14px; padding: 22px 12px; text-align: center; margin-top: 14px;
               font-size: 1.9rem; font-weight: 800; letter-spacing: 2px; }
    .verdict.yes { background: #e3f6ea; color: #17753a !important; border: 2px solid #5cc487; }
    .verdict.no  { background: #fde8e8; color: #a32727 !important; border: 2px solid #ee8c8c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Recreate the notebook's LabelEncoders in memory ----------
pclass_label_encoder = LabelEncoder().fit(np.ravel(pd.DataFrame({"Pclass": ["First", "Second", "Third"]})))
gender_label_encoder = LabelEncoder().fit(np.ravel(pd.DataFrame({"Gender": ["Male", "Female"]})))
sibling_label_encoder = LabelEncoder().fit(np.ravel(pd.DataFrame({"Sibling": ["Zero", "One", "Two", "Three"]})))
embarked_label_encoder = LabelEncoder().fit(np.ravel(pd.DataFrame({"Embarked": ["Southampton", "Cherbourg", "Queenstown"]})))


@st.cache_resource
def load_model():
    with open(Path(__file__).parent / "svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------- Page ----------
st.markdown(
    """
    <div class="hero">
        <h1>⚓ Titanic Survival Predictor</h1>
        <p>Choose a passenger's details and let the model predict whether they survived.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.markdown('<div class="q"><b>1</b>Ticket class</div>', unsafe_allow_html=True)
        pclass_input = st.radio("Ticket class (PClass)", ["First", "Second", "Third"],
                                horizontal=True, label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="q"><b>2</b>Gender</div>', unsafe_allow_html=True)
        gender_input = st.radio("Gender", ["Male", "Female"],
                                horizontal=True, label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="q"><b>3</b>Siblings / spouses aboard</div>', unsafe_allow_html=True)
        sibling_input = st.radio("Siblings / spouses aboard (Sibling)", ["Zero", "One", "Two", "Three"],
                                 horizontal=True, label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="q"><b>4</b>Port of embarkation</div>', unsafe_allow_html=True)
        embarked_input = st.radio("Port of embarkation (Embarked)", ["Southampton", "Cherbourg", "Queenstown"],
                                  horizontal=True, label_visibility="collapsed")

with right:
    st.markdown(
        f"""
        <div class="summary">
            <h4>Passenger summary</h4>
            <div class="row"><span>Class</span><span>{pclass_input}</span></div>
            <div class="row"><span>Gender</span><span>{gender_input}</span></div>
            <div class="row"><span>Siblings / spouses</span><span>{sibling_input}</span></div>
            <div class="row"><span>Embarked</span><span>{embarked_input}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Predict survival"):
        # Feature vector in the training order: PClass, Gender, Sibling, Embarked
        features = pd.DataFrame({
            "PClass": [pclass_label_encoder.transform([pclass_input])[0]],
            "Gender": [gender_label_encoder.transform([gender_input])[0]],
            "Sibling": [sibling_label_encoder.transform([sibling_input])[0]],
            "Embarked": [embarked_label_encoder.transform([embarked_input])[0]],
        })

        if model.predict(features)[0] == 1:
            st.markdown('<div class="verdict yes">SURVIVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict no">NOT SURVIVED</div>', unsafe_allow_html=True)

with st.expander("About the model"):
    st.write(
        "A Support Vector Classifier trained on 100 sample records using four inputs: "
        "ticket class, gender, siblings/spouses aboard and port of embarkation. "
        "Accuracy on the 20 held-out records: 0.80."
    )
    st.caption("AIC354 · Machine Learning Fundamentals · COMSATS Lahore. For learning only.")
