import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_advanced_model(input_shape=(224, 224, 3), fine_tune=False):
    """
    Builds an advanced model using ResNet50V2 transfer learning.
    """
    base_model = applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    if fine_tune:
        # Fine-tune from this layer onwards
        base_model.trainable = True
        # Let's freeze the first 100 layers
        for layer in base_model.layers[:100]:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

if __name__ == "__main__":
    model = build_advanced_model()
    model.summary()
