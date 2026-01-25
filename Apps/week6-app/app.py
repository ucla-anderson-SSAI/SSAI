"""
Flask backend for Week 6 CNN Training App
Real CNN training on CIFAR-10 with Keras
With queueing system for concurrent user limits
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import os
import time
from collections import deque
from datetime import datetime, timedelta
import threading
import uuid

# Set TensorFlow to CPU only and reduce logging BEFORE importing
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================
MAX_CONCURRENT_TRAININGS = 5  # Max simultaneous training jobs
SESSION_TIMEOUT_MINUTES = 10  # Clean up sessions after this long
CLEANUP_INTERVAL_SECONDS = 60  # How often to run cleanup

# ============================================
# Global State
# ============================================
training_sessions = {}  # session_id -> session data
training_queue = deque()  # Queue of session_ids waiting to train
active_trainings = set()  # Set of session_ids currently training
state_lock = threading.Lock()  # Thread-safe access to shared state

# Lazy-loaded data
_tf_loaded = False
_cifar_loaded = False
_X_TRAIN_FULL = None
_Y_TRAIN_FULL = None
_X_TEST = None
_Y_TEST = None
_tf = None
_keras = None
_layers = None

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def load_tensorflow():
    """Lazy load TensorFlow"""
    global _tf_loaded, _tf, _keras, _layers
    if _tf_loaded:
        return

    print("Loading TensorFlow...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _tf = tf
    _keras = keras
    _layers = layers
    _tf_loaded = True
    print(f"TensorFlow {tf.__version__} loaded")


def load_cifar():
    """Lazy load CIFAR-10 dataset"""
    global _cifar_loaded, _X_TRAIN_FULL, _Y_TRAIN_FULL, _X_TEST, _Y_TEST
    if _cifar_loaded:
        return

    load_tensorflow()

    print("Loading CIFAR-10 dataset...")
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    _X_TRAIN_FULL = X_train.astype('float32') / 255.0
    _X_TEST = X_test.astype('float32') / 255.0
    _Y_TRAIN_FULL = y_train.flatten()
    _Y_TEST = y_test.flatten()
    _cifar_loaded = True
    print(f"Loaded {len(_X_TRAIN_FULL)} training images, {len(_X_TEST)} test images")


# ============================================
# Session Cleanup Thread
# ============================================
def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT_MINUTES"""
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        now = datetime.now()

        with state_lock:
            expired = []
            for session_id, session in training_sessions.items():
                created = session.get('created_at', now)
                if now - created > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    expired.append(session_id)

            for session_id in expired:
                if session_id in active_trainings:
                    active_trainings.discard(session_id)
                if session_id in training_queue:
                    training_queue.remove(session_id)
                if training_sessions[session_id].get('model'):
                    del training_sessions[session_id]['model']
                del training_sessions[session_id]

            if expired:
                print(f"Cleaned up {len(expired)} expired sessions")


# ============================================
# Queue Processor Thread
# ============================================
def process_queue():
    """Process training queue - start jobs when slots available"""
    while True:
        time.sleep(0.5)

        with state_lock:
            while len(active_trainings) < MAX_CONCURRENT_TRAININGS and training_queue:
                session_id = training_queue.popleft()

                if session_id not in training_sessions:
                    continue

                session = training_sessions[session_id]

                if session['status'] not in ['queued']:
                    continue

                active_trainings.add(session_id)
                session['status'] = 'starting'

                thread = threading.Thread(
                    target=train_model_async,
                    args=(session_id, session['config'])
                )
                thread.daemon = True
                thread.start()


# ============================================
# Model Building and Training
# ============================================
def build_cnn_model(config):
    """Build CNN model based on frontend config"""
    model = _keras.Sequential()

    for i in range(config['convBlocks']):
        filters = config['filters'] * (2 ** min(i, 2))

        if i == 0:
            model.add(_layers.Conv2D(
                filters,
                (config['kernelSize'], config['kernelSize']),
                padding='same',
                activation='relu',
                input_shape=(32, 32, 3),
                name=f'conv2d_{i}'
            ))
        else:
            model.add(_layers.Conv2D(
                filters,
                (config['kernelSize'], config['kernelSize']),
                padding='same',
                activation='relu',
                name=f'conv2d_{i}'
            ))

        if config.get('batchNorm', False):
            model.add(_layers.BatchNormalization(name=f'bn_{i}'))

        model.add(_layers.MaxPooling2D(pool_size=(2, 2), name=f'pool_{i}'))

        if config.get('dropout', 0) > 0:
            model.add(_layers.Dropout(config['dropout']))

    model.add(_layers.Flatten())
    model.add(_layers.Dense(128, activation='relu'))
    model.add(_layers.Dropout(0.5))
    model.add(_layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=_keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def extract_filters(model, num_blocks):
    """Extract filter weights from trained model"""
    filters = {}

    for i in range(num_blocks):
        try:
            layer = model.get_layer(f'conv2d_{i}')
            weights = layer.get_weights()[0]

            h, w, in_c, out_f = weights.shape
            layer_filters = []

            for f in range(out_f):
                filter_data = []
                for y in range(h):
                    row = []
                    for x in range(w):
                        channels = [float(weights[y, x, c, f]) for c in range(in_c)]
                        row.append(channels)
                    filter_data.append(row)
                layer_filters.append(filter_data)

            filters[f'layer{i + 1}'] = layer_filters
        except Exception as e:
            print(f"Error extracting layer {i}: {e}")

    return filters


def train_model_async(session_id, config):
    """Train model in background thread"""
    session = training_sessions.get(session_id)
    if not session:
        return

    try:
        # Ensure data is loaded
        load_cifar()

        num_samples = config.get('numSamples', 1000)
        session['status'] = f'Preparing {num_samples} training samples...'

        samples_per_class = num_samples // 10
        indices = []
        for class_idx in range(10):
            class_indices = np.where(_Y_TRAIN_FULL == class_idx)[0]
            selected = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)
            indices.extend(selected)

        np.random.shuffle(indices)
        X_train = _X_TRAIN_FULL[indices]
        y_train = _Y_TRAIN_FULL[indices]

        test_indices = np.random.choice(len(_X_TEST), size=min(1000, len(_X_TEST)), replace=False)
        X_val = _X_TEST[test_indices]
        y_val = _Y_TEST[test_indices]

        session['status'] = 'Building model...'
        model = build_cnn_model(config)

        session['status'] = 'Training...'
        session['history'] = []
        session['current_epoch'] = 0

        class ProgressCallback(_keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                session['current_epoch'] = epoch + 1
                session['history'].append({
                    'trainAcc': float(logs['accuracy'] * 100),
                    'valAcc': float(logs['val_accuracy'] * 100),
                    'trainLoss': float(logs['loss']),
                    'valLoss': float(logs['val_loss'])
                })
                session['status'] = f"Epoch {epoch + 1}/{config['epochs']}"

        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[ProgressCallback()],
            verbose=0
        )

        session['status'] = 'Extracting learned filters...'
        filters = extract_filters(model, config['convBlocks'])
        session['filters'] = filters

        session['status'] = 'Evaluating...'
        test_loss, test_acc = model.evaluate(_X_TEST, _Y_TEST, verbose=0)
        session['test_accuracy'] = float(test_acc * 100)

        sample_indices = [np.where(_Y_TEST == i)[0][0] for i in range(10)][:8]
        sample_images = _X_TEST[sample_indices]
        predictions = model.predict(sample_images, verbose=0)

        session['sample_predictions'] = [
            {
                'true': int(_Y_TEST[sample_indices[i]]),
                'predicted': int(np.argmax(predictions[i])),
                'confidence': float(np.max(predictions[i]))
            }
            for i in range(len(sample_indices))
        ]

        session['status'] = 'complete'
        session['model'] = model

    except Exception as e:
        import traceback
        session['status'] = 'error'
        session['error'] = str(e)
        print(f"Training error: {traceback.format_exc()}")

    finally:
        with state_lock:
            active_trainings.discard(session_id)


# ============================================
# API Endpoints
# ============================================
@app.route('/api/health', methods=['GET'])
def health():
    """Health check - responds immediately without loading TF/CIFAR"""
    with state_lock:
        queue_length = len(training_queue)
        active_count = len(active_trainings)
        total_sessions = len(training_sessions)

    return jsonify({
        'status': 'ok',
        'tf_loaded': _tf_loaded,
        'cifar_loaded': _cifar_loaded,
        'queue_length': queue_length,
        'active_trainings': active_count,
        'max_concurrent': MAX_CONCURRENT_TRAININGS,
        'total_sessions': total_sessions
    })


@app.route('/api/warmup', methods=['POST'])
def warmup():
    """Pre-load TensorFlow and CIFAR-10"""
    load_cifar()
    return jsonify({
        'status': 'ok',
        'tf_version': _tf.__version__,
        'cifar_samples': len(_X_TRAIN_FULL)
    })


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start a new training session (may be queued)"""
    config = request.json
    session_id = str(uuid.uuid4())

    with state_lock:
        training_sessions[session_id] = {
            'status': 'queued',
            'config': config,
            'current_epoch': 0,
            'history': [],
            'filters': None,
            'model': None,
            'test_accuracy': None,
            'sample_predictions': None,
            'created_at': datetime.now(),
            'queue_position': len(training_queue) + 1
        }

        training_queue.append(session_id)
        queue_position = len(training_queue)
        active_count = len(active_trainings)

    return jsonify({
        'session_id': session_id,
        'queue_position': queue_position,
        'active_trainings': active_count,
        'max_concurrent': MAX_CONCURRENT_TRAININGS
    })


@app.route('/api/train/<session_id>', methods=['GET'])
def get_training_status(session_id):
    """Get training status and results"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]

    with state_lock:
        if session_id in training_queue:
            queue_position = list(training_queue).index(session_id) + 1
        else:
            queue_position = 0
        active_count = len(active_trainings)

    response = {
        'status': session['status'],
        'current_epoch': session['current_epoch'],
        'total_epochs': session['config']['epochs'],
        'history': session['history'],
        'queue_position': queue_position,
        'active_trainings': active_count,
        'max_concurrent': MAX_CONCURRENT_TRAININGS
    }

    if session['status'] == 'complete':
        response['filters'] = session['filters']
        response['test_accuracy'] = session['test_accuracy']
        response['sample_predictions'] = session['sample_predictions']

    if session.get('error'):
        response['error'] = session['error']

    return jsonify(response)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with trained model"""
    data = request.json
    session_id = data.get('session_id')

    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]
    if session.get('model') is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    image_data = data.get('image')

    if isinstance(image_data, list):
        img_array = np.array(image_data).reshape(1, 32, 32, 3) / 255.0
    else:
        return jsonify({'error': 'Invalid image format'}), 400

    prediction = session['model'].predict(img_array, verbose=0)[0]

    return jsonify({
        'predictions': [
            {'class': CIFAR10_CLASSES[i], 'probability': float(prediction[i])}
            for i in range(10)
        ],
        'top_class': CIFAR10_CLASSES[int(np.argmax(prediction))],
        'confidence': float(np.max(prediction))
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get server statistics"""
    with state_lock:
        return jsonify({
            'queue_length': len(training_queue),
            'active_trainings': len(active_trainings),
            'max_concurrent': MAX_CONCURRENT_TRAININGS,
            'total_sessions': len(training_sessions),
            'queued_sessions': [sid for sid in training_queue],
            'active_sessions': list(active_trainings)
        })


@app.route('/', methods=['GET'])
def root():
    """Serve the frontend"""
    return send_file('index.html')


# Start background threads
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    print(f"Max concurrent trainings: {MAX_CONCURRENT_TRAININGS}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
