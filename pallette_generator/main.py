import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
from base64 import b64encode
from matplotlib.figure import Figure

app = Flask(__name__)
model = load_model('pallete_generator.h5')

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def extract_average_face_color(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau format tidak didukung.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        raise ValueError("Tidak ada wajah yang terdeteksi dalam gambar.")
    
    x, y, w, h = faces[0]
    face_region = image[y:y+h, x:x+w]
    face_tensor = tf.convert_to_tensor(face_region, dtype=tf.float32)
    mean_color = tf.reduce_mean(face_tensor, axis=[0, 1])
    mean_color = mean_color.numpy().astype(int)
    hex_color = "#{:02x}{:02x}{:02x}".format(mean_color[2], mean_color[1], mean_color[0])
    
    return hex_color

def predict_palette(image_path):
    skin_tone_hex = extract_average_face_color(image_path)
    skin_tone_rgb = np.array([hex_to_rgb(skin_tone_hex)])
    
    predicted_palette = model.predict(skin_tone_rgb)
    predicted_palette = predicted_palette.reshape(-1, 3).astype(int)
    predicted_palette_hex = [rgb_to_hex(color) for color in predicted_palette]
    
    return skin_tone_hex, predicted_palette_hex

def display_color_palette(skin_tone_hex, predicted_colors):
    skin_tone_rgb = hex_to_rgb(skin_tone_hex)
    colors = [skin_tone_rgb] + [hex_to_rgb(color) for color in predicted_colors]
    color_labels = ["Extracted Skin Tone"] + [f"Color {i+1}" for i in range(len(predicted_colors))]
    
    fig, ax = plt.subplots(1, len(colors), figsize=(12, 2))
    for i, (color, label) in enumerate(zip(colors, color_labels)):
        ax[i].imshow([[color]])
        ax[i].set_title(label, fontsize=8)
        ax[i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return b64encode(buf.getvalue()).decode('ascii')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'static/' + file.filename
            file.save(file_path)
            skin_tone_hex, predicted_palette = predict_palette(file_path)
            img_data = display_color_palette(skin_tone_hex, predicted_palette)
            return render_template('index.html', img_data=img_data, skin_tone=skin_tone_hex, palette=predicted_palette)
    return render_template('index.html', img_data=None, skin_tone=None, palette=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
