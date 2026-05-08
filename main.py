import os
import argparse
import tensorflow as tf
from src.data_loading import get_data_generators, get_class_weights
from src.train_baseline import build_baseline_model
from src.train_advanced import build_advanced_model
from src.evaluate import plot_history, plot_confusion_matrix, get_metrics_report
from src.predict import predict_disease
import numpy as np

def train_and_evaluate(model_type='baseline', data_dir='data/raw/chest_xray/chest_xray', epochs=10, batch_size=32):
    print(f"Starting {model_type} model training...")
    print(f"Using data directory: {os.path.abspath(data_dir)}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    # Check for subdirectories
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"Error: Split directory {split_path} not found.")
            return
        else:
            print(f"Found {split} directory at {split_path}")

    # Data Loading
    train_gen, val_gen, test_gen = get_data_generators(data_dir, batch_size=batch_size)
    class_weights = get_class_weights(train_gen)
    
    # Model Selection
    if model_type == 'baseline':
        model = build_baseline_model()
    else:
        model = build_advanced_model(fine_tune=True)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint(f'models/{model_type}_best.h5', save_best_only=True)
    ]
    
    # Training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Evaluation
    print(f"Evaluating {model_type} model...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save Plots
    plot_history(history, f'outputs/plots/{model_type}_history.png')
    
    # Confusion Matrix
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        classes=list(test_gen.class_indices.keys()), 
        output_path=f'outputs/confusion_matrices/{model_type}_cm.png'
    )
    
    # Report
    report = get_metrics_report(y_true, y_pred, classes=list(test_gen.class_indices.keys()))
    with open(f'outputs/reports/{model_type}_report.txt', 'w') as f:
        f.write(report)
    
    print(f"Finished {model_type} training and evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Image Disease Classification')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'advanced'], help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='data/raw/chest_xray/chest_xray', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    train_and_evaluate(model_type=args.model, data_dir=args.data_dir, epochs=args.epochs)
