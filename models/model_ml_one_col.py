import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("../data/data_clean_one_col.csv")

def preprocessing(text):
    doc = nlp(text)
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    text_pre = " ".join(clean_tokens)
    return text_pre


X = data.drop("IsHate", axis=1)
y = data["IsHate"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocessing text
X_train["Text"] = X_train["Text"].apply(preprocessing)
X_train.head()

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train["Text"]).toarray()


# Inicializar y entrenar el modelo
model_nb = MultinomialNB()
model_nb.fit(X_train_vect, y_train)

X_test["Text"] = X_test["Text"].apply(preprocessing)
X_test = vectorizer.transform(X_test["Text"]).toarray()

# Predecir en el conjunto de prueba
y_pred = model_nb.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
