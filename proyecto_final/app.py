from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image
import re
import nltk
import os

# Descargar recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Función para limpiar texto
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

# Función para extraer texto de una imagen
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Función para procesar una imagen
def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Cargar el modelo entrenado
model = load_model('model_empresa_classification_compuesto.h5')

# Configurar el tokenizer (asegúrate de usar el mismo tokenizer que utilizaste para entrenar el modelo)
max_words = 10000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_words)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Procesar la imagen
    image_features = process_image(file_path)

    # Extraer y limpiar texto de la imagen
    extracted_text = extract_text_from_image(file_path)
    cleaned_text = clean_text(extracted_text)
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_data = pad_sequences(text_sequence, maxlen=max_sequence_length)

    # Verificar y ajustar las dimensiones de la imagen y el texto
    image_features = np.squeeze(image_features, axis=0)
    image_features = np.expand_dims(image_features, axis=0)  # Asegurar que sea (1, 224, 224, 3)

    # Hacer predicciones usando el modelo
    predictions = model.predict([image_features, text_data])
    predicted_class = np.argmax(predictions, axis=1)

    # Mapear el índice predicho a la etiqueta real
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = ['bantel perú','bitel','claro','directv' , 'entel', 'movistar'] 
    label_encoder.fit(labels)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return jsonify({"predicted_label": predicted_label[0], "extracted_text": extracted_text})

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
