import streamlit as st
import pandas as pd
import joblib
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError 

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.model_ml_one_col import TextPreprocessor

load_dotenv()

API_KEY = os.getenv("KEY_YT")
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_comments(video_id: str, max_results=30):
    try:
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)
    except HttpError:
        st.error(f"Ingrese una url valida.")
    return comments

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
    st.title("Is a Hater?")

    model, preprocess = load_models()

    option = st.sidebar.selectbox(
        "Escoge un modelo",
        ("Model ML", "Model DL", "Video de YouTube")
    )

    if option == "Model ML":
        st.header("Modelo Naive Bayes")
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
    elif option == "Model DL":
        st.header("Modelo de Red Neuronal")
        text = st.text_area("Ingresa un texto")
    elif option == "Video de YouTube":
        st.header("Comentarios de YouTube")
        video_id = st.text_input("Ingresa e lD del video de YouTube:")
        if st.button("Extraer"):
            if video_id:
                st.info("Extrayendo comentarios...")
                comments = get_comments(video_id)
                if comments:
                    for c in comments:
                        st.write(c)
                else:
                    st.write("No se encontraron comentarios")
            else:
                st.warning("Ingrese un ID valido.")


if __name__ == "__main__":
    main()