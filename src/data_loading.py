import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, target_size=(224, 224), batch_size=32, augmentation=True):
    """
    Creates data generators for train, validation, and test sets.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Data augmentation for training
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    # Only rescaling for validation and test
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def get_class_weights(train_generator):
    """
    Calculates class weights to handle imbalance.
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = train_generator.classes
    class_indices = train_generator.class_indices
    unique_classes = np.unique(classes)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=classes
    )
    
    return dict(zip(unique_classes, weights))
