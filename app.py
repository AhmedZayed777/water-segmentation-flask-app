from flask import Flask, render_template, request, send_file
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import cv2 # type: ignore
import tifffile as tiff
from matplotlib.figure import Figure

app=Flask(__name__)

model = load_model('model.h5', compile=False)

# === Preprocessing Function ===
def normalize_per_channel(image, band_indices):
    selected_bands = image[:, :, band_indices]
    norm_image = np.zeros_like(selected_bands, dtype=np.float32)
    for i in range(selected_bands.shape[2]):
        channel = selected_bands[:, :, i]
        c_min = channel.min()
        c_max = channel.max()
        norm_image[:, :, i] = (channel - c_min) / (c_max - c_min) if c_max - c_min != 0 else 0
    return norm_image

def compute_ndwi(image,green_idx,nir_idx):
    green=image[:,:,green_idx].astype(np.float32)
    nir=image[:,:,nir_idx].astype(np.float32)
    ndwi=(green - nir) / (green + nir + 1e-6)
    return ndwi[..., np.newaxis]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            print("Before loading image")
            img = tiff.imread(file)
            print("Image loaded", img.shape)

            # Band indices: adjust these based on what you trained on
            band_indices = [2, 3, 4, 5, 6,7]
            img_norm = normalize_per_channel(img, band_indices)
            ndwi = compute_ndwi(img, green_idx=1, nir_idx=3)
            print("NDWI", ndwi.shape)
            # Combine and resize to 224x224
            img_input = np.concatenate([img_norm, ndwi], axis=-1)

            # Resize to match model input size (224, 224)
            resized = cv2.resize(img_input, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_input = np.expand_dims(resized, axis=0)
            print("Input to model", img_input.shape)
            # Now: img_input.shape == (1, 224, 224, 7)
            pred_mask = model.predict(img_input)[0, :, :, 0]
            print("Prediction done")
            mask_img = (pred_mask > 0.5).astype(np.uint8) * 255

            # Convert prediction to base64 to show in browser
            fig = Figure()
            ax = fig.subplots()
            ax.imshow(mask_img, cmap='gray')
            ax.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()

            return render_template('index.html', result=img_b64)

    return render_template('index.html')





if __name__ == "__main__":
    app.run(debug=True,port=8000)