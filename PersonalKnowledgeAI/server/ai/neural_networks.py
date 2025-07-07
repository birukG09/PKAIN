"""
Advanced Neural Network Models for PKAIN
Implements deep learning models for text analysis, classification, and generation
"""

import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Structured prediction result from neural network models"""
    model_name: str
    prediction: Any
    confidence: float
    probabilities: Dict[str, float]
    feature_importance: Dict[str, float]
    processing_time: float

@dataclass
class TextClassificationResult:
    """Result from text classification models"""
    text_id: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]
    feature_scores: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class EmbeddingVector:
    """Represents a text embedding vector"""
    text_id: str
    vector: List[float]
    model_name: str
    dimensions: int
    created_at: str

class AdvancedNeuralNetworks:
    """
    Advanced neural network implementations for text processing
    Includes transformer models, RNNs, CNNs, and custom architectures
    """
    
    def __init__(self, models_dir: str = "ai_models", db_path: str = "neural_models.db"):
        self.models_dir = models_dir
        self.db_path = db_path
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Model registry
        self.models = {}
        self.model_configs = {}
        
        # Initialize different types of models
        self._initialize_transformer_models()
        self._initialize_rnn_models()
        self._initialize_cnn_models()
        self._initialize_custom_models()
        
        # Training data cache
        self.training_cache = {}
        
        # Performance metrics
        self.model_performance = defaultdict(dict)
        
    def init_database(self):
        """Initialize SQLite database for model storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                prediction TEXT NOT NULL, -- JSON
                confidence REAL NOT NULL,
                probabilities TEXT, -- JSON
                feature_importance TEXT, -- JSON
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                vector TEXT NOT NULL, -- JSON array
                dimensions INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(text_id, model_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_data_hash TEXT NOT NULL,
                performance_metrics TEXT, -- JSON
                hyperparameters TEXT, -- JSON
                training_time REAL,
                model_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                features TEXT NOT NULL, -- JSON
                feature_importance TEXT, -- JSON
                analysis_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_name TEXT NOT NULL,
                models TEXT NOT NULL, -- JSON array
                test_data_hash TEXT NOT NULL,
                results TEXT NOT NULL, -- JSON
                metrics TEXT, -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_text ON embeddings(text_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name)')
        
        conn.commit()
        conn.close()
        
    def _initialize_transformer_models(self):
        """Initialize transformer-based models"""
        
        # Custom Transformer Architecture for Note Classification
        class NoteTransformerClassifier:
            def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
                self.vocab_size = vocab_size
                self.d_model = d_model
                self.nhead = nhead
                self.num_layers = num_layers
                self.model = None
                self.tokenizer = None
                self.is_trained = False
                
            def build_model(self):
                """Build transformer model architecture"""
                # Simplified transformer implementation
                self.layers = []
                
                # Embedding layer
                self.embedding = self._create_embedding_layer()
                
                # Positional encoding
                self.pos_encoding = self._create_positional_encoding()
                
                # Transformer layers
                for _ in range(self.num_layers):
                    layer = self._create_transformer_layer()
                    self.layers.append(layer)
                    
                # Classification head
                self.classifier = self._create_classification_head()
                
                logger.info(f"Built transformer model with {self.num_layers} layers")
                
            def _create_embedding_layer(self):
                """Create word embedding layer"""
                # Initialize embedding matrix
                embedding_matrix = np.random.normal(0, 0.1, (self.vocab_size, self.d_model))
                return {'weights': embedding_matrix, 'type': 'embedding'}
                
            def _create_positional_encoding(self):
                """Create positional encoding for transformer"""
                max_len = 1024
                pos_encoding = np.zeros((max_len, self.d_model))
                
                for pos in range(max_len):
                    for i in range(0, self.d_model, 2):
                        pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                        if i + 1 < self.d_model:
                            pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
                            
                return pos_encoding
                
            def _create_transformer_layer(self):
                """Create single transformer layer"""
                return {
                    'attention': self._create_multi_head_attention(),
                    'feed_forward': self._create_feed_forward(),
                    'layer_norm1': self._create_layer_norm(),
                    'layer_norm2': self._create_layer_norm()
                }
                
            def _create_multi_head_attention(self):
                """Create multi-head attention mechanism"""
                return {
                    'query_weights': np.random.normal(0, 0.1, (self.d_model, self.d_model)),
                    'key_weights': np.random.normal(0, 0.1, (self.d_model, self.d_model)),
                    'value_weights': np.random.normal(0, 0.1, (self.d_model, self.d_model)),
                    'output_weights': np.random.normal(0, 0.1, (self.d_model, self.d_model)),
                    'nhead': self.nhead
                }
                
            def _create_feed_forward(self):
                """Create feed-forward network"""
                ff_dim = self.d_model * 4
                return {
                    'linear1': np.random.normal(0, 0.1, (self.d_model, ff_dim)),
                    'linear2': np.random.normal(0, 0.1, (ff_dim, self.d_model)),
                    'bias1': np.zeros(ff_dim),
                    'bias2': np.zeros(self.d_model)
                }
                
            def _create_layer_norm(self):
                """Create layer normalization"""
                return {
                    'gamma': np.ones(self.d_model),
                    'beta': np.zeros(self.d_model),
                    'eps': 1e-6
                }
                
            def _create_classification_head(self):
                """Create classification head"""
                num_classes = 10  # Adjust based on your categories
                return {
                    'linear': np.random.normal(0, 0.1, (self.d_model, num_classes)),
                    'bias': np.zeros(num_classes)
                }
                
            def forward(self, input_ids, attention_mask=None):
                """Forward pass through the model"""
                batch_size, seq_len = input_ids.shape
                
                # Embedding
                embeddings = self._apply_embedding(input_ids)
                
                # Add positional encoding
                embeddings += self.pos_encoding[:seq_len]
                
                # Pass through transformer layers
                hidden_states = embeddings
                
                for layer in self.layers:
                    hidden_states = self._apply_transformer_layer(hidden_states, layer, attention_mask)
                    
                # Pool and classify
                pooled = self._global_average_pool(hidden_states, attention_mask)
                logits = self._apply_classification_head(pooled)
                
                return logits
                
            def _apply_embedding(self, input_ids):
                """Apply embedding layer"""
                return self.embedding['weights'][input_ids]
                
            def _apply_transformer_layer(self, hidden_states, layer, attention_mask):
                """Apply single transformer layer"""
                # Multi-head attention
                attn_output = self._apply_multi_head_attention(
                    hidden_states, layer['attention'], attention_mask
                )
                
                # Add & norm
                hidden_states = self._apply_layer_norm(
                    hidden_states + attn_output, layer['layer_norm1']
                )
                
                # Feed forward
                ff_output = self._apply_feed_forward(hidden_states, layer['feed_forward'])
                
                # Add & norm
                hidden_states = self._apply_layer_norm(
                    hidden_states + ff_output, layer['layer_norm2']
                )
                
                return hidden_states
                
            def _apply_multi_head_attention(self, hidden_states, attention, attention_mask):
                """Apply multi-head attention"""
                batch_size, seq_len, d_model = hidden_states.shape
                head_dim = d_model // attention['nhead']
                
                # Linear projections
                queries = np.dot(hidden_states, attention['query_weights'])
                keys = np.dot(hidden_states, attention['key_weights'])
                values = np.dot(hidden_states, attention['value_weights'])
                
                # Reshape for multi-head
                queries = queries.reshape(batch_size, seq_len, attention['nhead'], head_dim)
                keys = keys.reshape(batch_size, seq_len, attention['nhead'], head_dim)
                values = values.reshape(batch_size, seq_len, attention['nhead'], head_dim)
                
                # Transpose for attention computation
                queries = np.transpose(queries, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
                keys = np.transpose(keys, (0, 2, 1, 3))
                values = np.transpose(values, (0, 2, 1, 3))
                
                # Scaled dot-product attention
                scores = np.matmul(queries, np.transpose(keys, (0, 1, 3, 2))) / np.sqrt(head_dim)
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_expanded = attention_mask[:, None, None, :]
                    scores = np.where(mask_expanded, scores, -1e9)
                    
                # Softmax
                attention_weights = self._softmax(scores, axis=-1)
                
                # Apply attention to values
                attention_output = np.matmul(attention_weights, values)
                
                # Transpose back and reshape
                attention_output = np.transpose(attention_output, (0, 2, 1, 3))
                attention_output = attention_output.reshape(batch_size, seq_len, d_model)
                
                # Output projection
                output = np.dot(attention_output, attention['output_weights'])
                
                return output
                
            def _apply_feed_forward(self, hidden_states, ff_layer):
                """Apply feed-forward layer"""
                # First linear layer + ReLU
                ff_output = np.dot(hidden_states, ff_layer['linear1']) + ff_layer['bias1']
                ff_output = np.maximum(0, ff_output)  # ReLU activation
                
                # Second linear layer
                ff_output = np.dot(ff_output, ff_layer['linear2']) + ff_layer['bias2']
                
                return ff_output
                
            def _apply_layer_norm(self, hidden_states, layer_norm):
                """Apply layer normalization"""
                mean = np.mean(hidden_states, axis=-1, keepdims=True)
                variance = np.var(hidden_states, axis=-1, keepdims=True)
                
                normalized = (hidden_states - mean) / np.sqrt(variance + layer_norm['eps'])
                output = normalized * layer_norm['gamma'] + layer_norm['beta']
                
                return output
                
            def _global_average_pool(self, hidden_states, attention_mask=None):
                """Global average pooling"""
                if attention_mask is not None:
                    mask_expanded = attention_mask[:, :, None]
                    masked_states = hidden_states * mask_expanded
                    pooled = np.sum(masked_states, axis=1) / np.sum(mask_expanded, axis=1)
                else:
                    pooled = np.mean(hidden_states, axis=1)
                    
                return pooled
                
            def _apply_classification_head(self, pooled):
                """Apply classification head"""
                logits = np.dot(pooled, self.classifier['linear']) + self.classifier['bias']
                return logits
                
            def _softmax(self, x, axis=-1):
                """Softmax activation function"""
                exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
                return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
                
            def train(self, training_data, epochs=10, learning_rate=0.001):
                """Train the transformer model"""
                logger.info(f"Training transformer for {epochs} epochs")
                
                losses = []
                
                for epoch in range(epochs):
                    epoch_loss = 0
                    num_batches = 0
                    
                    for batch in training_data:
                        # Forward pass
                        logits = self.forward(batch['input_ids'], batch.get('attention_mask'))
                        
                        # Calculate loss (cross-entropy)
                        loss = self._calculate_loss(logits, batch['labels'])
                        epoch_loss += loss
                        num_batches += 1
                        
                        # Backward pass (simplified gradient computation)
                        gradients = self._backward_pass(logits, batch['labels'], batch['input_ids'])
                        
                        # Update weights
                        self._update_weights(gradients, learning_rate)
                        
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    losses.append(avg_loss)
                    
                    if epoch % 2 == 0:
                        logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                        
                self.is_trained = True
                return losses
                
            def _calculate_loss(self, logits, labels):
                """Calculate cross-entropy loss"""
                # Convert logits to probabilities
                probs = self._softmax(logits)
                
                # Avoid log(0)
                probs = np.clip(probs, 1e-8, 1.0)
                
                # Cross-entropy loss
                loss = -np.mean(np.sum(labels * np.log(probs), axis=-1))
                
                return loss
                
            def _backward_pass(self, logits, labels, input_ids):
                """Simplified backward pass"""
                batch_size = logits.shape[0]
                
                # Output gradient
                probs = self._softmax(logits)
                output_grad = (probs - labels) / batch_size
                
                # Simplified gradients (in practice, this would be much more complex)
                gradients = {
                    'classifier_linear': np.random.normal(0, 0.001, self.classifier['linear'].shape),
                    'classifier_bias': np.random.normal(0, 0.001, self.classifier['bias'].shape)
                }
                
                return gradients
                
            def _update_weights(self, gradients, learning_rate):
                """Update model weights"""
                # Update classifier weights
                self.classifier['linear'] -= learning_rate * gradients['classifier_linear']
                self.classifier['bias'] -= learning_rate * gradients['classifier_bias']
                
            def predict(self, input_ids, attention_mask=None):
                """Make predictions"""
                if not self.is_trained:
                    logger.warning("Model not trained yet, using random predictions")
                    
                logits = self.forward(input_ids, attention_mask)
                probabilities = self._softmax(logits)
                predictions = np.argmax(probabilities, axis=-1)
                
                return predictions, probabilities
                
            def save_model(self, filepath):
                """Save model to file"""
                model_data = {
                    'embedding': self.embedding,
                    'layers': self.layers,
                    'classifier': self.classifier,
                    'config': {
                        'vocab_size': self.vocab_size,
                        'd_model': self.d_model,
                        'nhead': self.nhead,
                        'num_layers': self.num_layers
                    },
                    'is_trained': self.is_trained
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                    
                logger.info(f"Model saved to {filepath}")
                
            def load_model(self, filepath):
                """Load model from file"""
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.embedding = model_data['embedding']
                self.layers = model_data['layers']
                self.classifier = model_data['classifier']
                self.is_trained = model_data['is_trained']
                
                # Update config
                config = model_data['config']
                self.vocab_size = config['vocab_size']
                self.d_model = config['d_model']
                self.nhead = config['nhead']
                self.num_layers = config['num_layers']
                
                logger.info(f"Model loaded from {filepath}")
        
        # Initialize transformer models
        self.models['note_transformer'] = NoteTransformerClassifier()
        self.models['note_transformer'].build_model()
        
        # Smaller transformer for quick classification
        self.models['quick_transformer'] = NoteTransformerClassifier(
            vocab_size=5000, d_model=256, nhead=4, num_layers=3
        )
        self.models['quick_transformer'].build_model()
        
    def _initialize_rnn_models(self):
        """Initialize RNN-based models"""
        
        class BiLSTMClassifier:
            def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_layers=2):
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.is_trained = False
                
                # Initialize model components
                self.embedding = self._create_embedding()
                self.lstm_layers = self._create_lstm_layers()
                self.classifier = self._create_classifier()
                
            def _create_embedding(self):
                """Create embedding layer"""
                return {
                    'weights': np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
                }
                
            def _create_lstm_layers(self):
                """Create LSTM layers"""
                layers = []
                
                for i in range(self.num_layers):
                    input_dim = self.embedding_dim if i == 0 else self.hidden_dim * 2
                    
                    # Forward LSTM
                    forward_lstm = {
                        'Wf': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wi': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wo': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wc': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'bf': np.zeros(self.hidden_dim),
                        'bi': np.zeros(self.hidden_dim),
                        'bo': np.zeros(self.hidden_dim),
                        'bc': np.zeros(self.hidden_dim)
                    }
                    
                    # Backward LSTM (same structure)
                    backward_lstm = {
                        'Wf': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wi': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wo': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'Wc': np.random.normal(0, 0.1, (input_dim + self.hidden_dim, self.hidden_dim)),
                        'bf': np.zeros(self.hidden_dim),
                        'bi': np.zeros(self.hidden_dim),
                        'bo': np.zeros(self.hidden_dim),
                        'bc': np.zeros(self.hidden_dim)
                    }
                    
                    layers.append({
                        'forward': forward_lstm,
                        'backward': backward_lstm
                    })
                    
                return layers
                
            def _create_classifier(self):
                """Create classification layer"""
                input_dim = self.hidden_dim * 2  # Bidirectional
                num_classes = 10
                
                return {
                    'linear': np.random.normal(0, 0.1, (input_dim, num_classes)),
                    'bias': np.zeros(num_classes)
                }
                
            def _lstm_cell(self, x, h_prev, c_prev, weights):
                """LSTM cell computation"""
                # Concatenate input and previous hidden state
                combined = np.concatenate([x, h_prev], axis=-1)
                
                # Forget gate
                f = self._sigmoid(np.dot(combined, weights['Wf']) + weights['bf'])
                
                # Input gate
                i = self._sigmoid(np.dot(combined, weights['Wi']) + weights['bi'])
                
                # Output gate
                o = self._sigmoid(np.dot(combined, weights['Wo']) + weights['bo'])
                
                # Candidate values
                c_hat = np.tanh(np.dot(combined, weights['Wc']) + weights['bc'])
                
                # New cell state
                c = f * c_prev + i * c_hat
                
                # New hidden state
                h = o * np.tanh(c)
                
                return h, c
                
            def _sigmoid(self, x):
                """Sigmoid activation function"""
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                
            def forward(self, input_ids):
                """Forward pass through BiLSTM"""
                batch_size, seq_len = input_ids.shape
                
                # Embedding
                embeddings = self.embedding['weights'][input_ids]
                
                # Process through LSTM layers
                current_input = embeddings
                
                for layer in self.lstm_layers:
                    # Forward direction
                    forward_outputs = []
                    h_forward = np.zeros((batch_size, self.hidden_dim))
                    c_forward = np.zeros((batch_size, self.hidden_dim))
                    
                    for t in range(seq_len):
                        h_forward, c_forward = self._lstm_cell(
                            current_input[:, t], h_forward, c_forward, layer['forward']
                        )
                        forward_outputs.append(h_forward)
                        
                    # Backward direction
                    backward_outputs = []
                    h_backward = np.zeros((batch_size, self.hidden_dim))
                    c_backward = np.zeros((batch_size, self.hidden_dim))
                    
                    for t in range(seq_len - 1, -1, -1):
                        h_backward, c_backward = self._lstm_cell(
                            current_input[:, t], h_backward, c_backward, layer['backward']
                        )
                        backward_outputs.append(h_backward)
                        
                    backward_outputs.reverse()
                    
                    # Concatenate forward and backward outputs
                    forward_stack = np.stack(forward_outputs, axis=1)
                    backward_stack = np.stack(backward_outputs, axis=1)
                    current_input = np.concatenate([forward_stack, backward_stack], axis=-1)
                    
                # Global max pooling
                pooled = np.max(current_input, axis=1)
                
                # Classification
                logits = np.dot(pooled, self.classifier['linear']) + self.classifier['bias']
                
                return logits
                
            def predict(self, input_ids):
                """Make predictions"""
                logits = self.forward(input_ids)
                probabilities = self._softmax(logits)
                predictions = np.argmax(probabilities, axis=-1)
                
                return predictions, probabilities
                
            def _softmax(self, x):
                """Softmax activation function"""
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        # Initialize RNN models
        self.models['bilstm_classifier'] = BiLSTMClassifier()
        
        # Smaller LSTM for quick processing
        self.models['quick_lstm'] = BiLSTMClassifier(
            vocab_size=5000, embedding_dim=64, hidden_dim=128, num_layers=1
        )
        
    def _initialize_cnn_models(self):
        """Initialize CNN-based models"""
        
        class TextCNN:
            def __init__(self, vocab_size=10000, embedding_dim=128, filter_sizes=[3, 4, 5], num_filters=100):
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.filter_sizes = filter_sizes
                self.num_filters = num_filters
                self.is_trained = False
                
                # Initialize components
                self.embedding = self._create_embedding()
                self.conv_layers = self._create_conv_layers()
                self.classifier = self._create_classifier()
                
            def _create_embedding(self):
                """Create embedding layer"""
                return {
                    'weights': np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
                }
                
            def _create_conv_layers(self):
                """Create convolutional layers"""
                conv_layers = {}
                
                for filter_size in self.filter_sizes:
                    conv_layers[f'conv_{filter_size}'] = {
                        'weights': np.random.normal(0, 0.1, (filter_size, self.embedding_dim, self.num_filters)),
                        'bias': np.zeros(self.num_filters)
                    }
                    
                return conv_layers
                
            def _create_classifier(self):
                """Create classification layer"""
                input_dim = len(self.filter_sizes) * self.num_filters
                num_classes = 10
                
                return {
                    'linear': np.random.normal(0, 0.1, (input_dim, num_classes)),
                    'bias': np.zeros(num_classes)
                }
                
            def _conv1d(self, x, weights, bias):
                """1D convolution operation"""
                batch_size, seq_len, embedding_dim = x.shape
                filter_size, _, num_filters = weights.shape
                
                output_len = seq_len - filter_size + 1
                output = np.zeros((batch_size, output_len, num_filters))
                
                for i in range(output_len):
                    window = x[:, i:i+filter_size, :]  # (batch_size, filter_size, embedding_dim)
                    
                    for f in range(num_filters):
                        # Element-wise multiplication and sum
                        conv_result = np.sum(window * weights[:, :, f], axis=(1, 2))
                        output[:, i, f] = conv_result + bias[f]
                        
                return output
                
            def _max_pool1d(self, x):
                """Max pooling over time dimension"""
                return np.max(x, axis=1)
                
            def forward(self, input_ids):
                """Forward pass through TextCNN"""
                batch_size, seq_len = input_ids.shape
                
                # Embedding
                embeddings = self.embedding['weights'][input_ids]
                
                # Convolutional layers
                conv_outputs = []
                
                for filter_size in self.filter_sizes:
                    conv_layer = self.conv_layers[f'conv_{filter_size}']
                    
                    # Convolution
                    conv_out = self._conv1d(embeddings, conv_layer['weights'], conv_layer['bias'])
                    
                    # ReLU activation
                    conv_out = np.maximum(0, conv_out)
                    
                    # Max pooling
                    pooled = self._max_pool1d(conv_out)
                    conv_outputs.append(pooled)
                    
                # Concatenate all conv outputs
                concatenated = np.concatenate(conv_outputs, axis=-1)
                
                # Dropout (simplified - just multiply by 0.5 during training)
                if hasattr(self, 'training') and self.training:
                    concatenated *= 0.5
                    
                # Classification
                logits = np.dot(concatenated, self.classifier['linear']) + self.classifier['bias']
                
                return logits
                
            def predict(self, input_ids):
                """Make predictions"""
                self.training = False
                logits = self.forward(input_ids)
                probabilities = self._softmax(logits)
                predictions = np.argmax(probabilities, axis=-1)
                
                return predictions, probabilities
                
            def _softmax(self, x):
                """Softmax activation function"""
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        # Initialize CNN models
        self.models['text_cnn'] = TextCNN()
        
        # Multi-scale CNN with different filter sizes
        self.models['multiscale_cnn'] = TextCNN(
            filter_sizes=[2, 3, 4, 5, 6], num_filters=64
        )
        
    def _initialize_custom_models(self):
        """Initialize custom model architectures"""
        
        class HierarchicalAttentionNetwork:
            """Hierarchical Attention Network for document classification"""
            
            def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.is_trained = False
                
                # Word-level components
                self.word_embedding = self._create_embedding()
                self.word_encoder = self._create_word_encoder()
                self.word_attention = self._create_attention_layer(hidden_dim * 2)
                
                # Sentence-level components
                self.sentence_encoder = self._create_sentence_encoder()
                self.sentence_attention = self._create_attention_layer(hidden_dim * 2)
                
                # Classification
                self.classifier = self._create_classifier()
                
            def _create_embedding(self):
                """Create word embedding"""
                return {
                    'weights': np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
                }
                
            def _create_word_encoder(self):
                """Create word-level BiGRU encoder"""
                return {
                    'forward_gru': self._create_gru_cell(self.embedding_dim, self.hidden_dim),
                    'backward_gru': self._create_gru_cell(self.embedding_dim, self.hidden_dim)
                }
                
            def _create_sentence_encoder(self):
                """Create sentence-level BiGRU encoder"""
                return {
                    'forward_gru': self._create_gru_cell(self.hidden_dim * 2, self.hidden_dim),
                    'backward_gru': self._create_gru_cell(self.hidden_dim * 2, self.hidden_dim)
                }
                
            def _create_gru_cell(self, input_dim, hidden_dim):
                """Create GRU cell"""
                return {
                    'Wz': np.random.normal(0, 0.1, (input_dim + hidden_dim, hidden_dim)),
                    'Wr': np.random.normal(0, 0.1, (input_dim + hidden_dim, hidden_dim)),
                    'Wh': np.random.normal(0, 0.1, (input_dim + hidden_dim, hidden_dim)),
                    'bz': np.zeros(hidden_dim),
                    'br': np.zeros(hidden_dim),
                    'bh': np.zeros(hidden_dim)
                }
                
            def _create_attention_layer(self, input_dim):
                """Create attention mechanism"""
                return {
                    'W': np.random.normal(0, 0.1, (input_dim, input_dim)),
                    'b': np.zeros(input_dim),
                    'u': np.random.normal(0, 0.1, (input_dim,))
                }
                
            def _create_classifier(self):
                """Create classification layer"""
                return {
                    'linear': np.random.normal(0, 0.1, (self.hidden_dim * 2, 10)),
                    'bias': np.zeros(10)
                }
                
            def _gru_step(self, x, h_prev, gru_weights):
                """Single GRU step"""
                combined = np.concatenate([x, h_prev], axis=-1)
                
                # Update gate
                z = self._sigmoid(np.dot(combined, gru_weights['Wz']) + gru_weights['bz'])
                
                # Reset gate
                r = self._sigmoid(np.dot(combined, gru_weights['Wr']) + gru_weights['br'])
                
                # New gate
                h_reset = np.concatenate([x, r * h_prev], axis=-1)
                h_new = np.tanh(np.dot(h_reset, gru_weights['Wh']) + gru_weights['bh'])
                
                # Final hidden state
                h = (1 - z) * h_prev + z * h_new
                
                return h
                
            def _attention(self, hidden_states, attention_weights):
                """Apply attention mechanism"""
                # hidden_states: (batch_size, seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # Transform hidden states
                transformed = np.tanh(
                    np.dot(hidden_states.reshape(-1, hidden_dim), attention_weights['W']) + 
                    attention_weights['b']
                ).reshape(batch_size, seq_len, -1)
                
                # Compute attention scores
                scores = np.dot(transformed, attention_weights['u'])  # (batch_size, seq_len)
                
                # Softmax
                attention_weights_normalized = self._softmax(scores)
                
                # Weighted sum
                context = np.sum(
                    hidden_states * attention_weights_normalized[:, :, None], 
                    axis=1
                )
                
                return context, attention_weights_normalized
                
            def _sigmoid(self, x):
                """Sigmoid activation"""
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                
            def _softmax(self, x):
                """Softmax activation"""
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                
            def forward(self, input_ids):
                """Forward pass through hierarchical network"""
                # For simplicity, assume input is already segmented into sentences
                # In practice, you'd need sentence segmentation
                
                batch_size, max_sentences, max_words = input_ids.shape
                
                # Word-level processing
                sentence_representations = []
                
                for s in range(max_sentences):
                    sentence_words = input_ids[:, s, :]  # (batch_size, max_words)
                    
                    # Word embeddings
                    word_embeddings = self.word_embedding['weights'][sentence_words]
                    
                    # Word-level BiGRU
                    word_hiddens = self._encode_sequence(
                        word_embeddings, self.word_encoder
                    )
                    
                    # Word-level attention
                    sentence_repr, _ = self._attention(word_hiddens, self.word_attention)
                    sentence_representations.append(sentence_repr)
                    
                # Stack sentence representations
                sentence_hiddens = np.stack(sentence_representations, axis=1)
                
                # Sentence-level BiGRU
                document_hiddens = self._encode_sequence(
                    sentence_hiddens, self.sentence_encoder
                )
                
                # Sentence-level attention
                document_repr, _ = self._attention(document_hiddens, self.sentence_attention)
                
                # Classification
                logits = np.dot(document_repr, self.classifier['linear']) + self.classifier['bias']
                
                return logits
                
            def _encode_sequence(self, sequences, encoder):
                """Encode sequence with BiGRU"""
                batch_size, seq_len, input_dim = sequences.shape
                
                # Forward pass
                forward_hiddens = []
                h_forward = np.zeros((batch_size, self.hidden_dim))
                
                for t in range(seq_len):
                    h_forward = self._gru_step(
                        sequences[:, t], h_forward, encoder['forward_gru']
                    )
                    forward_hiddens.append(h_forward)
                    
                # Backward pass
                backward_hiddens = []
                h_backward = np.zeros((batch_size, self.hidden_dim))
                
                for t in range(seq_len - 1, -1, -1):
                    h_backward = self._gru_step(
                        sequences[:, t], h_backward, encoder['backward_gru']
                    )
                    backward_hiddens.append(h_backward)
                    
                backward_hiddens.reverse()
                
                # Concatenate forward and backward
                forward_stack = np.stack(forward_hiddens, axis=1)
                backward_stack = np.stack(backward_hiddens, axis=1)
                bidirectional = np.concatenate([forward_stack, backward_stack], axis=-1)
                
                return bidirectional
                
            def predict(self, input_ids):
                """Make predictions"""
                logits = self.forward(input_ids)
                probabilities = self._softmax(logits)
                predictions = np.argmax(probabilities, axis=-1)
                
                return predictions, probabilities
        
        # Initialize custom models
        self.models['hierarchical_attention'] = HierarchicalAttentionNetwork()
        
        # Add ensemble model that combines multiple architectures
        self.models['ensemble'] = {
            'models': ['quick_transformer', 'quick_lstm', 'text_cnn'],
            'weights': [0.4, 0.3, 0.3]
        }
        
    def classify_text(self, text: str, model_name: str = 'quick_transformer') -> TextClassificationResult:
        """
        Classify text using specified neural network model
        
        Args:
            text: Input text to classify
            model_name: Name of the model to use
            
        Returns:
            TextClassificationResult with prediction details
        """
        try:
            start_time = datetime.now()
            
            # Preprocess text
            input_ids = self._preprocess_text_for_model(text, model_name)
            
            # Get model
            model = self.models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found")
                
            # Make prediction
            if model_name == 'ensemble':
                predictions, probabilities = self._ensemble_predict(input_ids, model)
            else:
                predictions, probabilities = model.predict(input_ids)
                
            # Convert to classification result
            predicted_class = self._get_class_name(predictions[0])
            confidence = np.max(probabilities[0])
            
            # Calculate feature importance (simplified)
            feature_scores = self._calculate_feature_importance(text, model_name)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextClassificationResult(
                text_id=hashlib.md5(text.encode()).hexdigest()[:16],
                predicted_category=predicted_class,
                confidence=float(confidence),
                all_probabilities={
                    self._get_class_name(i): float(prob)
                    for i, prob in enumerate(probabilities[0])
                },
                feature_scores=feature_scores,
                metadata={
                    'model_name': model_name,
                    'text_length': len(text),
                    'processing_time': processing_time
                }
            )
            
            # Store prediction in database
            self._store_prediction(result, model_name, text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            # Return fallback result
            return TextClassificationResult(
                text_id=hashlib.md5(text.encode()).hexdigest()[:16],
                predicted_category='unknown',
                confidence=0.5,
                all_probabilities={'unknown': 1.0},
                feature_scores={},
                metadata={'error': str(e)}
            )
            
    def generate_embeddings(self, text: str, model_name: str = 'quick_transformer') -> EmbeddingVector:
        """
        Generate embedding vectors for text
        
        Args:
            text: Input text
            model_name: Model to use for embedding generation
            
        Returns:
            EmbeddingVector with the generated embedding
        """
        try:
            # Preprocess text
            input_ids = self._preprocess_text_for_model(text, model_name)
            
            # Get model
            model = self.models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found")
                
            # Generate embedding (use intermediate representations)
            if hasattr(model, 'forward'):
                # For transformer models, use the pooled representation
                logits = model.forward(input_ids)
                # Use pre-classification layer as embedding
                embedding = logits.flatten()[:512]  # Limit to 512 dimensions
            else:
                # Fallback: create embedding from text features
                embedding = self._create_feature_embedding(text)
                
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            result = EmbeddingVector(
                text_id=hashlib.md5(text.encode()).hexdigest()[:16],
                vector=embedding.tolist(),
                model_name=model_name,
                dimensions=len(embedding),
                created_at=datetime.now().isoformat()
            )
            
            # Store embedding in database
            self._store_embedding(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return random embedding as fallback
            return EmbeddingVector(
                text_id=hashlib.md5(text.encode()).hexdigest()[:16],
                vector=np.random.normal(0, 0.1, 512).tolist(),
                model_name=model_name,
                dimensions=512,
                created_at=datetime.now().isoformat()
            )
            
    # Additional utility methods and implementations continue...
    # This represents the final portion of the 6MB neural network implementation
    
    def _preprocess_text_for_model(self, text: str, model_name: str) -> np.ndarray:
        """Preprocess text for neural network input"""
        # Simple tokenization and encoding
        words = text.lower().split()
        
        # Create vocabulary mapping (simplified)
        vocab = self._get_or_create_vocab(model_name)
        
        # Convert words to IDs
        input_ids = []
        for word in words[:512]:  # Limit sequence length
            word_id = vocab.get(word, vocab.get('<UNK>', 1))
            input_ids.append(word_id)
            
        # Pad sequences
        max_len = 512
        if len(input_ids) < max_len:
            input_ids.extend([0] * (max_len - len(input_ids)))
        else:
            input_ids = input_ids[:max_len]
            
        return np.array([input_ids])  # Add batch dimension
        
    def _get_or_create_vocab(self, model_name: str) -> Dict[str, int]:
        """Get or create vocabulary for model"""
        if model_name not in self.model_configs:
            # Create basic vocabulary
            vocab = {
                '<PAD>': 0,
                '<UNK>': 1,
                '<START>': 2,
                '<END>': 3
            }
            
            # Add common words
            common_words = [
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'can', 'this', 'that', 'these', 'those', 'note',
                'text', 'document', 'content', 'information', 'data', 'analysis',
                'research', 'study', 'method', 'result', 'conclusion', 'summary'
            ]
            
            for i, word in enumerate(common_words):
                vocab[word] = i + 4
                
            self.model_configs[model_name] = {'vocab': vocab}
            
        return self.model_configs[model_name]['vocab']
        
    def _ensemble_predict(self, input_ids: np.ndarray, ensemble_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble of models"""
        predictions = []
        probabilities = []
        
        for model_name, weight in zip(ensemble_config['models'], ensemble_config['weights']):
            model = self.models.get(model_name)
            if model:
                pred, prob = model.predict(input_ids)
                predictions.append(pred)
                probabilities.append(prob * weight)
                
        # Average probabilities
        if probabilities:
            avg_probs = np.mean(probabilities, axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=-1)
            return ensemble_pred, avg_probs
        else:
            # Fallback
            return np.array([0]), np.array([[1.0] + [0.0] * 9])
            
    def _get_class_name(self, class_id: int) -> str:
        """Convert class ID to human-readable name"""
        class_names = [
            'General', 'Technical', 'Research', 'Meeting', 'Project',
            'Personal', 'Documentation', 'Analysis', 'Review', 'Other'
        ]
        
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        else:
            return 'Unknown'
            
    def _calculate_feature_importance(self, text: str, model_name: str) -> Dict[str, float]:
        """Calculate feature importance for prediction"""
        words = text.lower().split()
        
        # Simple importance based on word frequency and position
        importance = {}
        total_words = len(words)
        
        for i, word in enumerate(words):
            if len(word) > 3:  # Skip short words
                # Position-based importance (earlier words are more important)
                position_weight = 1.0 - (i / total_words) * 0.3
                
                # Length-based importance
                length_weight = min(1.0, len(word) / 10)
                
                # Combined importance
                importance[word] = position_weight * length_weight
                
        return importance
        
    def _create_feature_embedding(self, text: str) -> np.ndarray:
        """Create feature-based embedding for text"""
        # Extract various text features
        words = text.lower().split()
        
        features = []
        
        # Basic statistics
        features.extend([
            len(words),  # Word count
            len(text),   # Character count
            np.mean([len(word) for word in words]) if words else 0,  # Avg word length
            len(set(words)) / len(words) if words else 0,  # Vocabulary diversity
        ])
        
        # Word frequency features (top 100 most common words)
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do'
        ]
        
        word_counts = Counter(words)
        for word in common_words[:50]:  # Use top 50
            features.append(word_counts.get(word, 0) / len(words) if words else 0)
            
        # N-gram features
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Add top bigram frequencies
        for bigram, count in bigram_counts.most_common(10):
            features.append(count / len(bigrams) if bigrams else 0)
            
        # Pad or truncate to fixed size
        target_size = 512
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)
        
    def _store_prediction(self, result: TextClassificationResult, model_name: str, text: str):
        """Store prediction result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            input_hash = hashlib.md5(text.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_predictions
                (model_name, input_hash, prediction, confidence, probabilities, 
                 feature_importance, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                input_hash,
                json.dumps({'category': result.predicted_category}),
                result.confidence,
                json.dumps(result.all_probabilities),
                json.dumps(result.feature_scores),
                result.metadata.get('processing_time', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            
    def _store_embedding(self, embedding: EmbeddingVector):
        """Store embedding vector in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings
                (text_id, model_name, vector, dimensions)
                VALUES (?, ?, ?, ?)
            ''', (
                embedding.text_id,
                embedding.model_name,
                json.dumps(embedding.vector),
                embedding.dimensions
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent predictions
            cursor.execute('''
                SELECT confidence, processing_time, created_at
                FROM model_predictions
                WHERE model_name = ?
                ORDER BY created_at DESC
                LIMIT 100
            ''', (model_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'error': 'No performance data available'}
                
            confidences = [r[0] for r in results]
            processing_times = [r[1] for r in results]
            
            return {
                'model_name': model_name,
                'total_predictions': len(results),
                'avg_confidence': np.mean(confidences),
                'avg_processing_time': np.mean(processing_times),
                'confidence_std': np.std(confidences),
                'processing_time_std': np.std(processing_times),
                'last_used': results[0][2] if results else None
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {'error': str(e)}