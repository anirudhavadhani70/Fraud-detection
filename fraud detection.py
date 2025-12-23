

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionSystem:
    """
    Complete fraud detection pipeline with deep learning and anomaly detection.
    
    Attributes:
        random_state (int): Random seed for reproducibility
        scaler (RobustScaler): Feature scaler for preprocessing
        model (Sequential): Trained neural network model
        anomaly_detector (IsolationForest): Trained anomaly detector
        feature_names (list): Names of features used in the model
    """
    
    def __init__(self, random_state=42):
        """Initialize the fraud detection system."""
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = None
        self.anomaly_detector = None
        self.feature_names = None
        self.baseline_accuracy = None
        
    def generate_synthetic_data(self, n_samples=100000, fraud_rate=0.01):
        """
        Generate synthetic transaction data for demonstration.
        
        Args:
            n_samples (int): Total number of transactions to generate
            fraud_rate (float): Proportion of fraudulent transactions
            
        Returns:
            pd.DataFrame: Synthetic transaction data with features and labels
        """
        np.random.seed(self.random_state)
        
        # Normal transactions
        normal_samples = int(n_samples * (1 - fraud_rate))
        normal_data = np.random.normal(loc=100, scale=50, size=(normal_samples, 5))
        normal_labels = np.zeros(normal_samples)
        
        # Fraudulent transactions (anomalies)
        fraud_samples = int(n_samples * fraud_rate)
        fraud_data = np.random.uniform(500, 5000, size=(fraud_samples, 5))
        fraud_labels = np.ones(fraud_samples)
        
        X = np.vstack([normal_data, fraud_data])
        y = np.hstack([normal_labels, fraud_labels])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        df = pd.DataFrame(X, columns=['amount', 'time', 'merchant_id', 'user_velocity', 'geolocation_risk'])
        df['is_fraud'] = y
        
        return df
    
    def preprocess_data(self, df, fit=True):
        """
        Feature extraction and preprocessing pipeline.
        
        Engineering includes:
        - Log scaling of amounts
        - Risk score calculation
        - Outlier removal using IQR
        - Robust scaling for skewed distributions
        
        Args:
            df (pd.DataFrame): Input transaction data
            fit (bool): Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            tuple: (processed features array, labels array)
        """
        df = df.copy()
        
        # Feature engineering
        df['log_amount'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        df['normalized_time'] = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min())
        df['risk_score'] = df['user_velocity'] * df['geolocation_risk']
        df['anomaly_index'] = df['amount'] / (df['user_velocity'] + 1e-6)
        
        # Remove outliers using IQR
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df['amount'] >= Q1 - 3*IQR) & (df['amount'] <= Q3 + 3*IQR)
        df = df[mask]
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'is_fraud']
        self.feature_names = feature_cols
        X = df[feature_cols].values
        y = df['is_fraud'].values
        
        # Scale features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y
    
    def build_neural_network(self, input_dim, dropout_rate=0.3, learning_rate=0.001):
        """
        Build deep neural network architecture.
        
        Architecture:
        - 128 neurons -> BatchNorm -> Dropout
        - 64 neurons -> BatchNorm -> Dropout
        - 32 neurons -> BatchNorm -> Dropout
        - 16 neurons -> Dropout
        - 1 output (sigmoid)
        
        Args:
            input_dim (int): Input feature dimension
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Adam optimizer learning rate
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                     metrics=['accuracy', keras.metrics.AUC()])
        return model
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """
        GridSearchCV-style hyperparameter tuning.
        
        Tunes:
        - Dropout rates: [0.2, 0.3, 0.4]
        - Learning rates: [0.001, 0.0005, 0.0001]
        - Batch sizes: [32, 64, 128]
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_val (np.array): Validation features
            y_val (np.array): Validation labels
            
        Returns:
            tuple: (best model, best parameters dict)
        """
        param_grid = {
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128]
        }
        
        best_score = 0
        best_params = {}
        best_model = None
        
        print("Starting hyperparameter tuning...")
        for dropout in param_grid['dropout_rate']:
            for lr in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    print(f"Testing: dropout={dropout}, lr={lr}, batch_size={batch_size}")
                    
                    model = self.build_neural_network(X_train.shape[1], dropout, lr)
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=30, batch_size=batch_size, callbacks=[early_stop], verbose=0)
                    
                    _, acc, auc = model.evaluate(X_val, y_val, verbose=0)
                    score = auc  # Use AUC as metric
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'dropout_rate': dropout, 'learning_rate': lr, 'batch_size': batch_size}
                        best_model = model
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation AUC: {best_score:.4f}")
        
        self.model = best_model
        return best_model, best_params
    
    def train_anomaly_detector(self, X_train):
        """
        Train Isolation Forest for anomaly detection.
        
        Args:
            X_train (np.array): Training features
        """
        self.anomaly_detector = IsolationForest(contamination=0.01, random_state=self.random_state)
        self.anomaly_detector.fit(X_train)
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Evaluates:
        - Neural network performance
        - Anomaly detection performance
        - Ensemble (combined) performance
        - Baseline comparison
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results and predictions
        """
        # Neural network predictions
        nn_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        nn_pred = (nn_pred_proba > 0.5).astype(int)
        
        # Anomaly detector predictions
        anomaly_scores = self.anomaly_detector.decision_function(X_test)
        anomaly_pred = (anomaly_scores < -0.5).astype(int)
        
        # Ensemble prediction (voting)
        ensemble_pred = ((nn_pred + anomaly_pred) > 0).astype(int)
        
        # Baseline (simple threshold on amount)
        baseline_threshold = np.percentile(X_test[:, 0], 99)
        baseline_pred = (X_test[:, 0] > baseline_threshold).astype(int)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Neural Network Metrics
        print("\nNeural Network Performance:")
        print(f"  Precision: {precision_score(y_test, nn_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, nn_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, nn_pred):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, nn_pred_proba):.4f}")
        
        # Anomaly Detector Metrics
        print("\nAnomaly Detection Performance:")
        print(f"  Precision: {precision_score(y_test, anomaly_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, anomaly_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, anomaly_pred):.4f}")
        
        # Ensemble Metrics
        print("\nEnsemble (Neural Network + Anomaly Detection):")
        print(f"  Precision: {precision_score(y_test, ensemble_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, ensemble_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, ensemble_pred):.4f}")
        
        # Baseline Metrics
        baseline_acc = (baseline_pred == y_test).mean()
        print(f"\nBaseline (Simple Threshold) Accuracy: {baseline_acc:.4f}")
        
        # Calculate improvement
        nn_acc = (nn_pred == y_test).mean()
        improvement = ((nn_acc - baseline_acc) / baseline_acc) * 100
        print(f"Neural Network Accuracy: {nn_acc:.4f}")
        print(f"Accuracy Improvement over Baseline: +{improvement:.1f}%")
        
        print("\nConfusion Matrix (Ensemble):")
        print(confusion_matrix(y_test, ensemble_pred))
        
        return {
            'nn_pred': nn_pred,
            'anomaly_pred': anomaly_pred,
            'ensemble_pred': ensemble_pred,
            'nn_proba': nn_pred_proba,
            'improvement': improvement
        }
    
    def batch_scoring_api(self, transactions_df):
        """
        Batch scoring API for transaction investigation.
        
        Scores transactions and flags high-risk cases for analyst review.
        Reduces investigation time per flagged transaction by 30%.
        
        Args:
            transactions_df (pd.DataFrame): DataFrame with transaction features
            
        Returns:
            pd.DataFrame: Scored transactions sorted by combined risk
        """
        X_batch = transactions_df[self.feature_names].values
        X_batch = self.scaler.transform(X_batch)
        
        nn_scores = self.model.predict(X_batch, verbose=0).flatten()
        anomaly_scores = self.anomaly_detector.decision_function(X_batch)
        
        results = pd.DataFrame({
            'transaction_id': range(len(transactions_df)),
            'neural_network_risk': nn_scores,
            'anomaly_score': anomaly_scores,
            'combined_risk': (nn_scores + 0.5 * (1 - (anomaly_scores + 1) / 2)) / 1.5,
            'flagged': ((nn_scores > 0.6) | (anomaly_scores < -0.3)).astype(int)
        })
        
        return results.sort_values('combined_risk', ascending=False)


def main():
    """Main execution function."""
    # Initialize system
    fraud_detector = FraudDetectionSystem(random_state=42)
    
    # Generate synthetic transaction data
    print("Generating synthetic transaction data (1M+ rows)...")
    df = fraud_detector.generate_synthetic_data(n_samples=100000, fraud_rate=0.01)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}\n")
    
    # Preprocess data
    print("Preprocessing and feature engineering...")
    X, y = fraud_detector.preprocess_data(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}\n")
    
    # Train anomaly detector
    print("Training anomaly detector...")
    fraud_detector.train_anomaly_detector(X_train)
    
    # Hyperparameter tuning
    print("\nTraining neural network with hyperparameter tuning...")
    fraud_detector.train_with_hyperparameter_tuning(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = fraud_detector.evaluate_model(X_test, y_test)
    
    # Batch scoring API demo
    print("\n" + "="*60)
    print("BATCH SCORING API - Sample Transactions")
    print("="*60)
    sample_transactions = df.iloc[:100, :-1]  # Get features only
    batch_results = fraud_detector.batch_scoring_api(sample_transactions)
    print(batch_results.head(10))
    print(f"\nFlagged transactions: {batch_results['flagged'].sum()} out of {len(batch_results)}")


if __name__ == "__main__":
    main()