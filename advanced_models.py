# --- File: advanced_models.py ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, LSTM, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# New model for spatiotemporal analysis
def build_cnn_lstm_model(input_shape, num_classes):
    """
    Builds a spatiotemporal model (CNN + LSTM).
    input_shape should be (sequence_length, height, width, channels).
    """
    model = tf.keras.Sequential()

    # Add a TimeDistributed wrapper to apply a CNN to each frame in the sequence
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))

    # Add an LSTM layer to process the temporal features
    model.add(LSTM(128, activation='tanh', return_sequences=False))

    # Add final classification layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

def build_vgg16_model(input_shape, num_classes):
    """Builds a model using VGG16 for transfer learning."""
    # VGG16 requires input images of at least 32x32 pixels, with 3 channels.
    # We will reshape our 48x48x1 images to 48x48x3.
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add new classification layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)