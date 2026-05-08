import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads and preprocesses a single image for inference.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(model, img_path, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Predicts the disease class and confidence score.
    """
    processed_img = load_and_preprocess_image(img_path)
    prediction = model.predict(processed_img)[0][0]
    
    # For binary classification (sigmoid)
    if prediction > 0.5:
        class_idx = 1
        confidence = prediction
    else:
        class_idx = 0
        confidence = 1 - prediction
        
    result = {
        'class': class_names[class_idx],
        'confidence': float(confidence),
        'raw_score': float(prediction)
    }
    return result
