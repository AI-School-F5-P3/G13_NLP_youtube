import re
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import spacy
import joblib
import mlflow

from typing import Tuple, Dict, List

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("../data/data_clean_one_col.csv")

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X["Text"].values
        else:
            texts = X
        return [self._preprocess_text(text) for text in texts]
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub("([^\x00-\x7F])+"," ",text)
        text = re.sub("\n", " ", text)
        text = re.sub(" +", " ", text)
        text = re.sub('[0-9]',"", text)
        doc = self.nlp(text)
        clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(clean_tokens)


def pipeline_text(max_features=5000):
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vectorizer', TfidfVectorizer(max_features=max_features))
    ])
    return pipeline


def split_data(data: pd.DataFrame):
    X = data.drop("IsHate", axis=1)
    y = data["IsHate"]

    return train_test_split(X, y, test_size=0.3, random_state=42)



# Preprocessing text
# X_train["Text"] = X_train["Text"].apply(preprocessing)
# X_train.head()

# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_vect = vectorizer.fit_transform(X_train["Text"]).toarray()


# Inicializar y entrenar el modelo
# model_nb = MultinomialNB()
# model_nb.fit(X_train_vect, y_train)

# X_test["Text"] = X_test["Text"].apply(preprocessing)
# X_test = vectorizer.transform(X_test["Text"]).toarray()

# Predecir en el conjunto de prueba
# y_pred = model_nb.predict(X_test)

# Evaluar el modelo
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# # Overfitting
# y_train_pred = model_nb.predict(X_train_vect)
# print("Overfitting: ", accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_pred))

def get_models() -> Dict[str, BaseEstimator]:
    return {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(random_state=42),
        'linear_svc': LinearSVC(random_state=42)
    }


def evaluate_model(y_true, y_pred, prefix):
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision'],
        f"{prefix}recall": classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall'],
        f"{prefix}f1": classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']
    }
    return metrics


def train_model():
    mlflow.set_experiment("hate_speech_youtube")

    X_train, X_test, y_train, y_test = split_data(data)

    # entrenar pipeline
    pipeline = pipeline_text()
    X_train_transformed = pipeline.fit_transform(X_train).toarray()
    X_test_transformed = pipeline.transform(X_test).toarray()

    joblib.dump(pipeline, "preprocessing_tfid.joblib")

    # models
    models = get_models()

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)

            model.fit(X_train_transformed, y_train)

            # Evaluar en conjunto de entrenamiento
            y_train_pred = model.predict(X_train_transformed)
            train_metrics = evaluate_model(y_train, y_train_pred, prefix="train_")
            
            # Evaluar en conjunto de prueba
            y_test_pred = model.predict(X_test_transformed)
            test_metrics = evaluate_model(y_test, y_test_pred, prefix="test_")
            
            # Calcular overfitting
            overfitting = train_metrics['train_accuracy'] - test_metrics['test_accuracy']

            # Registrar métricas en MLflow
            for metrics in [train_metrics, test_metrics]:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metric('overfitting', overfitting)

            joblib.dump(model, f'model_{model_name}.joblib')
            mlflow.log_artifact(f'model_{model_name}.joblib')
            mlflow.log_artifact("preprocessing_tfid.joblib")

train_model()