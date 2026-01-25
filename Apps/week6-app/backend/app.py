"""
Flask backend for Week 6 CNN Training App
Provides real CNN training on CIFAR-10 with Keras/TensorFlow
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import threading
import uuid
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Store training sessions
training_sessions = {}

def build_cnn_model(config):
    """Build CNN model based on config"""
    model = keras.Sequential()
    input_channels = 3

    for i in range(config['convBlocks']):
        filters = config['filters'] * (2 ** min(i, 2))

        if i == 0:
            model.add(layers.Conv2D(
                filters,
                config['kernelSize'],
                padding='same',
                activation='relu',
                input_shape=(32, 32, 3),
                name=f'conv2d_{i}'
            ))
        else:
            model.add(layers.Conv2D(
                filters,
                config['kernelSize'],
                padding='same',
                activation='relu',
                name=f'conv2d_{i}'
            ))

        if config.get('batchNorm', False):
            model.add(layers.BatchNormalization(name=f'bn_{i}'))

        model.add(layers.MaxPooling2D(pool_size=2, name=f'pool_{i}'))

        if config.get('dropout', 0) > 0:
            model.add(layers.Dropout(config['dropout']))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def extract_filters(model, num_blocks):
    """Extract filter weights from trained model"""
    filters = {}
    for i in range(num_blocks):
        try:
            conv_layer = model.get_layer(f'conv2d_{i}')
            weights = conv_layer.get_weights()[0]  # [H, W, in_channels, out_channels]
            filters[f'layer{i+1}'] = weights.tolist()
        except:
            pass
    return filters

def filter_to_image(weights, layer_idx):
    """Convert filter weights to base64 image for visualization"""
    # weights shape: [H, W, in_channels, out_channels]
    images = []
    num_filters = min(8, weights.shape[-1])

    for f in range(num_filters):
        if layer_idx == 0 and weights.shape[2] == 3:
            # First layer with RGB - show as color
            filter_data = weights[:, :, :, f]
            # Normalize to 0-255
            filter_data = (filter_data - filter_data.min()) / (filter_data.max() - filter_data.min() + 1e-8)
            filter_data = (filter_data * 255).astype(np.uint8)
        else:
            # Average across input channels for grayscale
            filter_data = weights[:, :, :, f].mean(axis=2)
            # Normalize to 0-255
            filter_data = (filter_data - filter_data.min()) / (filter_data.max() - filter_data.min() + 1e-8)
            filter_data = (filter_data * 255).astype(np.uint8)
            filter_data = np.stack([filter_data] * 3, axis=-1)

        images.append(filter_data.tolist())

    return images

def train_model_async(session_id, config):
    """Train model in background thread"""
    session = training_sessions[session_id]

    try:
        # Load CIFAR-10
        session['status'] = 'Loading CIFAR-10 dataset...'
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # One-hot encode
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        # Use subset for faster training (still much more than 20 images!)
        # Use 10,000 training samples for reasonable speed
        train_size = min(10000, len(x_train))
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]

        session['status'] = 'Building model...'
        model = build_cnn_model(config)

        session['status'] = 'Training...'
        session['history'] = []

        # Custom callback to update progress
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                session['current_epoch'] = epoch + 1
                session['history'].append({
                    'trainAcc': float(logs['accuracy'] * 100),
                    'valAcc': float(logs['val_accuracy'] * 100),
                    'trainLoss': float(logs['loss']),
                    'valLoss': float(logs['val_loss'])
                })
                session['status'] = f"Training epoch {epoch + 1}/{config['epochs']}..."

        # Train
        model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=config['epochs'],
            validation_split=0.2,
            callbacks=[ProgressCallback()],
            verbose=0
        )

        session['status'] = 'Extracting filters...'

        # Extract filter weights
        filters = extract_filters(model, config['convBlocks'])
        session['filters'] = filters

        # Convert filters to images for easy visualization
        filter_images = {}
        for i in range(config['convBlocks']):
            try:
                conv_layer = model.get_layer(f'conv2d_{i}')
                weights = conv_layer.get_weights()[0]
                filter_images[f'layer{i+1}'] = filter_to_image(weights, i)
            except:
                pass
        session['filter_images'] = filter_images

        # Evaluate on test set
        session['status'] = 'Evaluating...'
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        session['test_accuracy'] = float(test_acc * 100)

        # Save some predictions for visualization
        predictions = model.predict(x_test[:8], verbose=0)
        session['sample_predictions'] = [
            {
                'true': int(y_test[i].argmax()),
                'predicted': int(predictions[i].argmax()),
                'confidence': float(predictions[i].max())
            }
            for i in range(8)
        ]

        session['status'] = 'complete'
        session['model'] = model

    except Exception as e:
        session['status'] = f'error: {str(e)}'
        session['error'] = str(e)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'tf_version': tf.__version__})

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start a new training session"""
    config = request.json
    session_id = str(uuid.uuid4())

    training_sessions[session_id] = {
        'status': 'initializing',
        'config': config,
        'current_epoch': 0,
        'history': [],
        'filters': None,
        'filter_images': None,
        'model': None
    }

    # Start training in background
    thread = threading.Thread(target=train_model_async, args=(session_id, config))
    thread.start()

    return jsonify({'session_id': session_id})

@app.route('/api/train/<session_id>', methods=['GET'])
def get_training_status(session_id):
    """Get training status and results"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]

    response = {
        'status': session['status'],
        'current_epoch': session['current_epoch'],
        'total_epochs': session['config']['epochs'],
        'history': session['history']
    }

    if session['status'] == 'complete':
        response['filters'] = session['filters']
        response['filter_images'] = session['filter_images']
        response['test_accuracy'] = session.get('test_accuracy')
        response['sample_predictions'] = session.get('sample_predictions')

    if 'error' in session:
        response['error'] = session['error']

    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with trained model"""
    data = request.json
    session_id = data.get('session_id')
    image_data = data.get('image')  # base64 or array

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]
    if session['model'] is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    # Process image
    if isinstance(image_data, str) and image_data.startswith('data:'):
        # Base64 image
        import base64
        from PIL import Image
        base64_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_bytes))
        img = img.resize((32, 32))
        img_array = np.array(img)[:, :, :3] / 255.0
    else:
        # Array
        img_array = np.array(image_data) / 255.0

    img_array = img_array.reshape(1, 32, 32, 3)

    prediction = session['model'].predict(img_array, verbose=0)[0]

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    return jsonify({
        'predictions': [
            {'class': classes[i], 'probability': float(prediction[i])}
            for i in range(10)
        ],
        'top_class': classes[int(prediction.argmax())],
        'confidence': float(prediction.max())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
