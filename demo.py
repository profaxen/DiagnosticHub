import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.predict import predict_disease, load_and_preprocess_image
from src.explainability import get_gradcam_heatmap, save_and_display_gradcam

def run_demo(data_dir='data/raw/chest_xray/chest_xray', model_dir='models'):
    print("--- Medical Disease Classification Demo ---")
    
    # 1. Select a random image from the test set (Pneumonia for better visualization)
    test_pneumonia_dir = os.path.join(data_dir, 'test', 'PNEUMONIA')
    test_normal_dir = os.path.join(data_dir, 'test', 'NORMAL')
    
    # Try to pick a pneumonia image first as it's more interesting for Grad-CAM
    if os.path.exists(test_pneumonia_dir):
        img_name = random.choice(os.listdir(test_pneumonia_dir))
        img_path = os.path.join(test_pneumonia_dir, img_name)
    else:
        print("Error: Test directory not found.")
        return

    print(f"Selected Image: {img_path}")

    # 2. Load Models
    print("Loading models...")
    baseline_model = tf.keras.models.load_model(os.path.join(model_dir, 'baseline_best.h5'))
    advanced_model = tf.keras.models.load_model(os.path.join(model_dir, 'advanced_best.h5'))

    # 3. Run Inference
    print("\n--- Inference Results ---")
    baseline_result = predict_disease(baseline_model, img_path)
    advanced_result = predict_disease(advanced_model, img_path)
    
    print(f"Baseline Model:  {baseline_result['class']} (Confidence: {baseline_result['confidence']:.2%})")
    print(f"Advanced Model:  {advanced_result['class']} (Confidence: {advanced_result['confidence']:.2%})")

    # 4. Generate Explainability (Grad-CAM) for Advanced Model
    print("\nGenerating Grad-CAM heatmap for Advanced Model...")
    img_array = load_and_preprocess_image(img_path)
    
    # Identify the last conv layer for ResNet50V2
    # For ResNet50V2, the last conv layer is often 'post_relu' or similar in the last block
    last_conv_layer_name = None
    for layer in reversed(advanced_model.get_layer('resnet50v2').layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if last_conv_layer_name:
        # Get the sub-model (ResNet50V2)
        resnet_model = advanced_model.get_layer('resnet50v2')
        
        # Create a model that outputs the last conv layer of the ResNet and the final output
        grad_model = tf.keras.models.Model(
            [advanced_model.inputs], 
            [resnet_model.get_layer(last_conv_layer_name).output, advanced_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads
        
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = np.dot(output, weights)
        
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        
        save_and_display_gradcam(img_path, heatmap, cam_path='outputs/plots/demo_gradcam.png')
        print("Grad-CAM visualization saved to: outputs/plots/demo_gradcam.png")
    else:
        print("Could not find conv layer for Grad-CAM.")

if __name__ == "__main__":
    print("Script started...")
    try:
        run_demo()
        print("Script finished successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
