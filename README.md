# **Clasificador de Comentarios de Odio en YouTube**

Este proyecto utiliza técnicas de procesamiento de lenguaje natural (NLP) para clasificar comentarios de YouTube en categorías de mensajes de odio. La solución utiliza un enfoque de Machine Learning.

---

## **Características Principales**
- **Detección de comentarios de odio**: Clasifica los comentarios.
- **Interfaz de usuario con Streamlit**: Una aplicación interactiva que permite cargar comentarios y predecirlos de un video de YouTube, además de obtener predicciones en tiempo real.
- **Modelos utilizados**:
  - SVM
  - Logistic Regression
  - Naive Bayes

## **Requisitos**
- Python 3.8 o superior.
- Instalar las dependencias necesarias con:
  ```bash
  pip install -r requirements.txt
  ```

## **Uso de la Aplicación**
- Ejecuta la aplicación Streamlit
  ```bash
  streamlit run app/main.py
  ```