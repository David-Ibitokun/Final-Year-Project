"""
Utility functions for loading and using models
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import json

class ModelPredictor:
    """Class to handle model loading and predictions"""
    
    def __init__(self, model_type="hybrid"):
        """
        Initialize predictor
        
        Args:
            model_type: One of ['cnn', 'gru', 'hybrid']
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self):
        """Load the trained model"""
        model_path = Path(f"models/{self.model_type}_model.keras")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def load_preprocessing_metadata(self):
        """Load preprocessing metadata"""
        metadata_path = Path("project_data/processed_data/preprocessing_metadata.json")
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        return None
    
    def prepare_input(self, input_data):
        """
        Prepare input data for prediction
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Preprocessed input array
        """
        # This is a placeholder - implement actual preprocessing
        # based on your training pipeline
        
        # Convert input_data to appropriate format
        # Apply same transformations as training
        
        return np.array([[0] * 20])  # Placeholder
    
    def predict(self, input_data):
        """
        Make prediction
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Predicted yield value
        """
        if self.model is None:
            self.load_model()
        
        # Prepare input
        X = self.prepare_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(X, verbose=0)
        
        return float(prediction[0][0])
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            self.load_model()
        
        return self.model.summary()

def load_all_models():
    """Load all three models"""
    models = {}
    
    for model_type in ['cnn', 'gru', 'hybrid']:
        try:
            predictor = ModelPredictor(model_type)
            predictor.load_model()
            models[model_type] = predictor
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            models[model_type] = None
    
    return models

def get_model_metrics():
    """Get performance metrics for all models"""
    # This would load actual metrics from training
    # Placeholder values for now
    
    metrics = {
        'cnn': {
            'mae': 0.45,
            'rmse': 0.62,
            'r2': 0.82,
            'training_time': 15
        },
        'gru': {
            'mae': 0.38,
            'rmse': 0.53,
            'r2': 0.87,
            'training_time': 22
        },
        'hybrid': {
            'mae': 0.32,
            'rmse': 0.47,
            'r2': 0.91,
            'training_time': 35
        }
    }
    
    return metrics
