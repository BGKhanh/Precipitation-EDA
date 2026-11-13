# =============================================================================
# NEURAL NETWORK MODELS: RNN, GRU, LSTM, BiLSTM
# =============================================================================

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Neural network models will not work.")

from .base import BaseRainfallModel

warnings.filterwarnings('ignore')


if TORCH_AVAILABLE:
    
    class RNNRainfallModel(BaseRainfallModel):
        """
        RNN model for rainfall prediction.
        Supports both single-stage and two-stage approaches.
        """
        
        def __init__(self,
                     use_two_stage: bool = True,
                     classification_threshold: float = 0.1,
                     sequence_length: int = 7,
                     hidden_size: int = 64,
                     num_layers: int = 2,
                     dropout: float = 0.2,
                     batch_size: int = 32,
                     epochs: int = 100,
                     learning_rate: float = 0.001,
                     random_state: int = 42):
            """
            Initialize RNN model.
            
            Args:
                use_two_stage: Whether to use two-stage approach
                classification_threshold: Threshold for rain/no-rain (mm/day)
                sequence_length: Length of input sequences
                hidden_size: Hidden layer size
                num_layers: Number of RNN layers
                dropout: Dropout rate
                batch_size: Training batch size
                epochs: Number of training epochs
                learning_rate: Learning rate
                random_state: Random seed
            """
            super().__init__(use_two_stage, classification_threshold, random_state)
            
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.batch_size = batch_size
            self.epochs = epochs
            self.learning_rate = learning_rate
            
            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            
        def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Create sequences for RNN input."""
            sequences = []
            targets = [] if y is not None else None
            
            for i in range(self.sequence_length, len(X)):
                seq = X.iloc[i-self.sequence_length:i].values
                sequences.append(seq)
                
                if y is not None:
                    targets.append(y.iloc[i])
            
            sequences = np.array(sequences)
            targets = np.array(targets) if targets is not None else None
            
            return sequences, targets
        
        def _create_classification_model(self, input_size: int, **kwargs):
            """Create RNN classifier."""
            return RNNNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='classification'
            )
        
        def _create_regression_model(self, input_size: int, **kwargs):
            """Create RNN regressor."""
            return RNNNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='regression'
            )
        
        def _create_single_stage_model(self, input_size: int, **kwargs):
            """Create single-stage RNN regressor."""
            return RNNNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='regression'
            )
        
        def _train_neural_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray, task: str):
            """Train neural network model."""
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Loss function and optimizer
            if task == 'classification':
                criterion = nn.BCELoss()
            else:
                criterion = nn.MSELoss()
                
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    outputs = model(batch_X)
                    if task == 'classification':
                        loss = criterion(outputs.squeeze(), batch_y)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        def _fit_two_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs):
            """Fit two-stage RNN model."""
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y)
            input_size = X_seq.shape[2]
            
            y_clf, y_reg = self._prepare_targets(pd.Series(y_seq))
            
            print(f"   📊 Stage 1: Classification ({y_clf.sum()} rain days / {len(y_clf)} total)")
            
            # Stage 1: Classification
            self.classification_model = self._create_classification_model(input_size)
            self._train_neural_model(self.classification_model, X_seq, y_clf.values, 'classification')
            
            # Stage 2: Regression (only on rainy days)
            rain_mask = y_clf == 1
            if rain_mask.sum() > 0:
                print(f"   📈 Stage 2: Regression on {rain_mask.sum()} rainy days")
                X_rain = X_seq[rain_mask]
                y_rain = y_reg[rain_mask]
                
                self.regression_model = self._create_regression_model(input_size)
                self._train_neural_model(self.regression_model, X_rain, y_rain.values, 'regression')
            else:
                print("   ⚠️ No rainy days found for regression training")
                self.regression_model = None
        
        def _fit_single_stage(self, X: pd.DataFrame, y: pd.Series, **kwargs):
            """Fit single-stage RNN model."""
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y)
            input_size = X_seq.shape[2]
            
            print(f"   📈 Single-stage regression on {len(y_seq)} sequences")
            self.single_stage_model = self._create_single_stage_model(input_size)
            self._train_neural_model(self.single_stage_model, X_seq, y_seq, 'regression')
        
        def _predict_two_stage(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            """Make two-stage predictions."""
            X_seq, _ = self._create_sequences(X)
            X_tensor = torch.FloatTensor(X_seq)
            
            # Classification predictions
            self.classification_model.eval()
            with torch.no_grad():
                clf_probs = self.classification_model(X_tensor).squeeze().numpy()
            
            # Regression predictions
            if self.regression_model is not None:
                self.regression_model.eval()
                with torch.no_grad():
                    reg_preds = self.regression_model(X_tensor).squeeze().numpy()
            else:
                reg_preds = np.zeros(len(X_seq))
            
            return clf_probs, reg_preds
        
        def _predict_single_stage(self, X: pd.DataFrame) -> np.ndarray:
            """Make single-stage predictions."""
            X_seq, _ = self._create_sequences(X)
            X_tensor = torch.FloatTensor(X_seq)
            
            self.single_stage_model.eval()
            with torch.no_grad():
                predictions = self.single_stage_model(X_tensor).squeeze().numpy()
            
            return predictions


    class RNNNet(nn.Module):
        """
        Basic RNN network for rainfall prediction.
        """
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                     output_size: int, dropout: float = 0.2, task: str = 'regression'):
            """
            Initialize RNN network.
            
            Args:
                input_size: Number of input features
                hidden_size: Hidden layer size
                num_layers: Number of RNN layers
                output_size: Number of output units
                dropout: Dropout rate
                task: 'classification' or 'regression'
            """
            super(RNNNet, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.task = task
            
            # RNN layer
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Output layer
            self.fc = nn.Linear(hidden_size, output_size)
            
            # Activation function
            if task == 'classification':
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.Identity()
        
        def forward(self, x):
            """Forward pass."""
            # RNN output
            rnn_out, _ = self.rnn(x)
            
            # Use last time step
            output = self.fc(rnn_out[:, -1, :])
            output = self.activation(output)
            
            return output


    class LSTMRainfallModel(RNNRainfallModel):
        """
        LSTM model for rainfall prediction.
        Inherits from RNNRainfallModel with LSTM-specific network.
        """
        
        def _create_classification_model(self, input_size: int, **kwargs):
            """Create LSTM classifier."""
            return LSTMNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='classification'
            )
        
        def _create_regression_model(self, input_size: int, **kwargs):
            """Create LSTM regressor."""
            return LSTMNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='regression'
            )
        
        def _create_single_stage_model(self, input_size: int, **kwargs):
            """Create single-stage LSTM regressor."""
            return LSTMNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
                task='regression'
            )


    class LSTMNet(nn.Module):
        """
        LSTM network for rainfall prediction.
        """
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     output_size: int, dropout: float = 0.2, task: str = 'regression'):
            """Initialize LSTM network."""
            super(LSTMNet, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.task = task
            
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Output layer
            self.fc = nn.Linear(hidden_size, output_size)
            
            # Activation function
            if task == 'classification':
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.Identity()
        
        def forward(self, x):
            """Forward pass."""
            # LSTM output
            lstm_out, _ = self.lstm(x)
            
            # Use last time step
            output = self.fc(lstm_out[:, -1, :])
            output = self.activation(output)
            
            return output

else:
    # Dummy classes when PyTorch is not available
    class RNNRainfallModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network models")
    
    class LSTMRainfallModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network models") 