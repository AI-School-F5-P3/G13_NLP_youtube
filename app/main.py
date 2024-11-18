import streamlit as st
import pandas as pd
import joblib


import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.model_ml_one_col import TextPreprocessor

def load_model(path):
    with open(path, 'rb') as file:
        model = joblib.load(file)
    return model


def load_pipeline(path):
    with open(path, 'rb') as file:
        pipeline = joblib.load(file)
    return pipeline

def load_models():
    model = load_model("../models/model_naive_bayes.joblib")
    preprocess = load_pipeline("../models/preprocessing_tfid.joblib")
    return model, preprocess

def predict_text(text, preprocess, model):
    if isinstance(text, str):
        text = pd.Series([text])
    text_processed = preprocess.transform(text)
    return model.predict(text_processed)[0]

def main():
    st.title("Clasificador de mensajes de odio.")

    model, preprocess = load_models()

    option = st.sidebar.selectbox(
        "Escoge un modelo",
        ("Model ML", "Model DL", "Video de youtube")
    )

    text = st.text_area("Ingresa un texto")
    if st.button("Clasificar"):
        if text:
            try:
                prediction = predict_text(text, preprocess, model)
                st.write(prediction)
            except Exception as e:
                st.error("Error: ", e)
        else:
            st.warning("Ingresa un texto.")
if __name__ == "__main__":
    main()