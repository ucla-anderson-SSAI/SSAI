"""
Flask backend for Week 2 CNN Training App
CNN training on CIFAR-100 subset: Lions, Tigers, and Bears!
OPTIMIZED for 80 concurrent students
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration - OPTIMIZED FOR 80 STUDENTS
# ============================================
MAX_CONCURRENT_TRAININGS = 15
SESSION_TIMEOUT_MINUTES = 15
CLEANUP_INTERVAL_SECONDS = 30

# Default training settings optimized for speed
DEFAULT_EPOCHS = 10
DEFAULT_NUM_SAMPLES = 500
MAX_EPOCHS = 15
MAX_SAMPLES = 1500  # Max is 500 per class * 3 classes

# ============================================
# CIFAR-100 Class Configuration
# Lions, Tigers, and Bears - Oh My!
# ============================================
CIFAR100_TARGET_CLASSES = {
    3: 'bear',    # CIFAR-100 index 3 -> our class 0
    43: 'lion',   # CIFAR-100 index 43 -> our class 1
    88: 'tiger'   # CIFAR-100 index 88 -> our class 2
}

# Our simplified class names (in order: 0, 1, 2)
CLASS_NAMES = ['bear', 'lion', 'tiger']
NUM_CLASSES = 3

# ============================================
# Global State
# ============================================
training_sessions = {}
training_queue = deque()
active_trainings = set()
state_lock = threading.Lock()

# Lazy-loaded data
_tf_loaded = False
_data_loaded = False
_X_TRAIN = None
_Y_TRAIN = None
_X_TEST = None
_Y_TEST = None
_tf = None
_keras = None
_layers = None


def load_tensorflow():
    """Lazy load TensorFlow"""
    global _tf_loaded, _tf, _keras, _layers
    if _tf_loaded:
        return

    print("Loading TensorFlow...")
    import tensorflow as tf
    
    # Limit TensorFlow memory and threads for better concurrent performance
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    from tensorflow import keras
    from tensorflow.keras import layers
    _tf = tf
    _keras = keras
    _layers = layers
    _tf_loaded = True
    print(f"TensorFlow {tf.__version__} loaded (optimized for concurrency)")


def load_data():
    """Lazy load CIFAR-100 and filter to lions, tigers, bears"""
    global _data_loaded, _X_TRAIN, _Y_TRAIN, _X_TEST, _Y_TEST
    if _data_loaded:
        return

    load_tensorflow()

    print("Loading CIFAR-100 dataset...")
    from tensorflow.keras.datasets import cifar100
    (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar100.load_data(label_mode='fine')
    
    y_train_full = y_train_full.flatten()
    y_test_full = y_test_full.flatten()
    
    # Filter to only our target classes
    target_indices = list(CIFAR100_TARGET_CLASSES.keys())
    
    # Training data
    train_mask = np.isin(y_train_full, target_indices)
    X_train_filtered = X_train_full[train_mask].astype('float32') / 255.0
    y_train_filtered = y_train_full[train_mask]
    
    # Test data
    test_mask = np.isin(y_test_full, target_indices)
    X_test_filtered = X_test_full[test_mask].astype('float32') / 255.0
    y_test_filtered = y_test_full[test_mask]
    
    # Remap labels: CIFAR-100 indices -> 0, 1, 2
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(target_indices))}
    # label_map: {3: 0, 43: 1, 88: 2} -> bear=0, lion=1, tiger=2
    
    _Y_TRAIN = np.array([label_map[y] for y in y_train_filtered])
    _Y_TEST = np.array([label_map[y] for y in y_test_filtered])
    _X_TRAIN = X_train_filtered
    _X_TEST = X_test_filtered
    
    _data_loaded = True
    
    # Print class distribution
    print(f"Loaded Lions, Tigers, and Bears dataset:")
    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(_Y_TRAIN == i)
        test_count = np.sum(_Y_TEST == i)
        print(f"  {name}: {train_count} train, {test_count} test")
    print(f"Total: {len(_X_TRAIN)} training, {len(_X_TEST)} test images")


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
        time.sleep(0.3)

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
    model.add(_layers.Dense(64, activation='relu'))
    model.add(_layers.Dropout(0.5))
    model.add(_layers.Dense(NUM_CLASSES, activation='softmax'))  # 3 classes now!

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

            # Only extract first 16 filters to reduce response size
            for f in range(min(out_f, 16)):
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
        load_data()

        # Apply caps to prevent abuse
        num_samples = min(config.get('numSamples', DEFAULT_NUM_SAMPLES), MAX_SAMPLES)
        epochs = min(config.get('epochs', DEFAULT_EPOCHS), MAX_EPOCHS)
        
        session['status'] = f'Preparing {num_samples} training samples...'

        # Sample balanced across 3 classes
        samples_per_class = num_samples // NUM_CLASSES
        indices = []
        for class_idx in range(NUM_CLASSES):
            class_indices = np.where(_Y_TRAIN == class_idx)[0]
            selected = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=False
            )
            indices.extend(selected)

        np.random.shuffle(indices)
        X_train = _X_TRAIN[indices]
        y_train = _Y_TRAIN[indices]

        # Validation set
        val_indices = np.random.choice(len(_X_TEST), size=min(300, len(_X_TEST)), replace=False)
        X_val = _X_TEST[val_indices]
        y_val = _Y_TEST[val_indices]

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
                session['status'] = f"Epoch {epoch + 1}/{epochs}"

        model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=epochs,
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

        # Sample predictions - get examples from each class
        sample_predictions = []
        for class_idx in range(NUM_CLASSES):
            class_test_indices = np.where(_Y_TEST == class_idx)[0]
            if len(class_test_indices) > 0:
                # Get 2-3 examples per class
                selected = np.random.choice(class_test_indices, size=min(3, len(class_test_indices)), replace=False)
                for idx in selected:
                    pred = model.predict(_X_TEST[idx:idx+1], verbose=0)[0]
                    sample_predictions.append({
                        'true': int(_Y_TEST[idx]),
                        'predicted': int(np.argmax(pred)),
                        'confidence': float(np.max(pred))
                    })

        session['sample_predictions'] = sample_predictions[:8]  # Limit to 8
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
    """Health check"""
    with state_lock:
        queue_length = len(training_queue)
        active_count = len(active_trainings)
        total_sessions = len(training_sessions)

    return jsonify({
        'status': 'ok',
        'tf_loaded': _tf_loaded,
        'data_loaded': _data_loaded,
        'queue_length': queue_length,
        'active_trainings': active_count,
        'max_concurrent': MAX_CONCURRENT_TRAININGS,
        'total_sessions': total_sessions,
        'dataset': 'Lions, Tigers, and Bears (CIFAR-100 subset)',
        'classes': CLASS_NAMES,
        'optimized_for': '80 students'
    })


@app.route('/api/warmup', methods=['POST'])
def warmup():
    """Pre-load TensorFlow and dataset"""
    load_data()
    return jsonify({
        'status': 'ok',
        'tf_version': _tf.__version__,
        'classes': CLASS_NAMES,
        'train_samples': len(_X_TRAIN),
        'test_samples': len(_X_TEST)
    })


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start a new training session"""
    config = request.json
    session_id = str(uuid.uuid4())
    
    config['epochs'] = min(config.get('epochs', DEFAULT_EPOCHS), MAX_EPOCHS)
    config['numSamples'] = min(config.get('numSamples', DEFAULT_NUM_SAMPLES), MAX_SAMPLES)

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

    estimated_wait = max(0, (queue_position - MAX_CONCURRENT_TRAININGS)) * 25

    return jsonify({
        'session_id': session_id,
        'queue_position': queue_position,
        'active_trainings': active_count,
        'max_concurrent': MAX_CONCURRENT_TRAININGS,
        'estimated_wait_seconds': estimated_wait
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
        'max_concurrent': MAX_CONCURRENT_TRAININGS,
        'classes': CLASS_NAMES  # Include class names for frontend
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
            {'class': CLASS_NAMES[i], 'probability': float(prediction[i])}
            for i in range(NUM_CLASSES)
        ],
        'top_class': CLASS_NAMES[int(np.argmax(prediction))],
        'confidence': float(np.max(prediction))
    })


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get the class names"""
    return jsonify({
        'classes': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'dataset': 'CIFAR-100 subset'
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
    print(f"Dataset: Lions, Tigers, and Bears (CIFAR-100 subset)")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Max concurrent trainings: {MAX_CONCURRENT_TRAININGS}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
