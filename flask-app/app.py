from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import joblib


model = load_model('model/model1_resized_1_all.h5')
scaler = joblib.load('scaler_resized_1_all.pkl')

# model = load_model('model/model1_resized_5_all.h5')
# scaler = joblib.load('scaler_resized_5_all.pkl')

# model = load_model('model/model2_resized_1_all.h5')
# scaler = joblib.load('scaler_resized_1_all.pkl')

# model = load_model('model/model2_resized_5_all.h5')
# scaler = joblib.load('scaler_resized_5_all.pkl')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 1024))
    sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, sharpen_kernel)
    img = cv2.Laplacian(img, cv2.CV_64F)
    img = cv2.convertScaleAbs(img)
    return img

def resize_image(image):
    img = cv2.resize(image, (512, 1024))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Konversi ke grayscale
    img = img_gray.astype('uint8')
    return img

def extract_features(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    # # Print GLCM shape
    # print("GLCM shape:", glcm.shape)
    # glcm_props = []
    # for name in props:
    #     prop = graycoprops(glcm, name)[0]
    #     print(f"Property: {name}, Values: {prop}")
    #     glcm_props.extend(prop)
    glcm_props = [property for name in props for property in graycoprops(glcm, name)[0]]
    return np.array(glcm_props)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename).replace('\\', '/')
            images = os.path.join('images', file.filename).replace('\\', '/')
            file.save(filename)

            image = cv2.imread(filename)
            
            # processed_image = preprocess_image(image)
            # print("Preprocessed Image Shape:", processed_image.shape)

            resized_image = resize_image(image)
            print("Resized Image Shape:", resized_image.shape)

            extractFeatures = extract_features(resized_image)
            print("Extracted Features:", extractFeatures)
            print("Extracted Shape:", extractFeatures.shape)

            reshapeFeatures = extractFeatures.reshape(1, -1)
            print("Reshape Features:", reshapeFeatures.shape)

            scalerFeatures = scaler.transform(reshapeFeatures)
            print("Scaler Features:", scalerFeatures)
            print("Scaler Shape:", scalerFeatures.shape)

            prediction = model.predict(scalerFeatures)
            print("Prediction:", prediction)

            class_prediction = 'Palsu' if prediction[0][0] > 0.5 else 'Asli'
            confidence = float(prediction[0][0]) if class_prediction == 'Palsu' else 1 - float(prediction[0][0])

            print("Class Prediction:", class_prediction)
            print("Confidence:", confidence)
            
            result = f"{class_prediction} (Confidence: {confidence:.2f})"
            
            return render_template('index.html', prediction=result, class_predict=class_prediction, filename=images)
    
    return render_template('index.html', prediction=None, filename=None)

if __name__ == '__main__':
    app.run(debug=True)