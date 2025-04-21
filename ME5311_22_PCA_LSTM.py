import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# Create results directories
os.makedirs('results_slp', exist_ok=True)
os.makedirs('results_t2m', exist_ok=True)
os.makedirs('results_slp_lstm', exist_ok=True)
os.makedirs('results_t2m_lstm', exist_ok=True)

# PyTorch Dataset Class for Sequences
class TimeSeriesDataset(Dataset):
    def __init__(self, X, seq_length):
        self.X = X
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x_seq = self.X[idx:idx+self.seq_length]
        # Target is the next time step
        y_target = self.X[idx+self.seq_length]
        
        return torch.FloatTensor(x_seq), torch.FloatTensor(y_target)

# PyTorch LSTM Model with three layers
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim3, output_dim)
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)  # lstm1_out shape: [batch, seq_len, hidden_dim1]
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # lstm2_out shape: [batch, seq_len, hidden_dim2]
        lstm2_out = self.dropout2(lstm2_out)

        # Third LSTM layer
        lstm3_out, _ = self.lstm3(lstm2_out)  # lstm3_out shape: [batch, seq_len, hidden_dim3]
        lstm3_out = self.dropout3(lstm3_out[:, -1, :])  # Take only the last time step output

        # Output layer
        output = self.fc(lstm3_out)
        return output

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, train_losses, val_losses

# Generate multi-step predictions
def generate_multi_step_predictions(model, initial_sequence, steps, device):
    model.eval()
    predictions = []
    
    # Convert initial sequence to tensor
    current_seq = torch.FloatTensor(initial_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(steps):
            # Predict next step
            next_step = model(current_seq).cpu().numpy()[0]
            predictions.append(next_step)
            
            # Update sequence for next prediction
            current_seq = torch.cat([
                current_seq[:, 1:, :],
                torch.FloatTensor(next_step).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
    
    return np.array(predictions)

# Function to apply FFT to PCA components
def apply_fft(X_pca, threshold_percent):
    X_pca_fft = np.zeros_like(X_pca)
    n_samples = X_pca.shape[0]
    for i in range(X_pca.shape[1]):
        freq = np.fft.rfft(X_pca[:, i])
        max_amp = np.max(np.abs(freq))
        threshold = max_amp * threshold_percent / 100.0
        freq[np.abs(freq) < threshold] = 0
        X_pca_fft[:, i] = np.fft.irfft(freq, n=n_samples)
    return X_pca_fft

# dimensions of data
n_samples = 16071
n_latitudes = 101 
n_longitudes = 161
shape = (n_samples, n_latitudes, n_longitudes)

# Data types to process
data_types = [
    {'file': 'data/slp.nc', 'var': 'msl', 'name': 'Sea Level Pressure', 'unit': 'Pa', 
     'dir': 'results_slp', 'lstm_dir': 'results_slp_lstm', 'n_components': 60},
    {'file': 'data/t2m.nc', 'var': 't2m', 'name': '2-meter Temperature', 'unit': 'K', 
     'dir': 'results_t2m', 'lstm_dir': 'results_t2m_lstm', 'n_components': 120}
]

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 对每种数据类型进行PCA分析
for data_config in data_types:
    print(f"\nprocess {data_config['name']}data...")
    
    # 加载数据
    ds = xr.open_dataset(data_config['file'])
    
    # 获取数据值
    da = ds[data_config['var']]
    x = da.values
    
    # 获取时间快照
    time = ds['time'].values
    
    # 获取经纬度值
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    
    # 重塑数据用于PCA: 从(n_samples, n_lat, n_lon)到(n_samples, n_features)
    X_reshaped = x.reshape(n_samples, n_latitudes * n_longitudes)
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # 执行PCA
    n_components = data_config['n_components']  # 从配置中获取主成分数量
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # # After performing PCA
    # X_reconstructed, mse = visualize_pca_results(
    #     x, X_pca, pca, scaler, time, lon, lat,
    #     n_samples, n_latitudes, n_longitudes, data_config
    # )

    # Apply FFT to PCA components
    threshold_percent = 0.1  # Set your threshold percentage here
    X_pca_fft = apply_fft(X_pca, threshold_percent)

    # =============== LSTM时间序列预测 ===============
    print(f"\n run {data_config['name']} 's LSTM time series prediction...")
    
    # Create LSTM results directory
    lstm_dir = data_config['lstm_dir']
    os.makedirs(lstm_dir, exist_ok=True)
    
    # 1. Split data into train/validation/test sets (7:2:1)
    n_total = X_pca_fft.shape[0]
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    
    X_train = X_pca_fft[:n_train]
    X_val = X_pca_fft[n_train:n_train+n_val]
    X_test = X_pca_fft[n_train+n_val:]
    
    print(f"Data split: Train {X_train.shape[0]} samples, Validation {X_val.shape[0]} samples, Test {X_test.shape[0]} samples")
    
    # 2. Create PyTorch datasets and dataloaders
    seq_length = 100  # Use past 30 days to predict next day
    batch_size = 64
    
    train_dataset = TimeSeriesDataset(X_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 3. Create LSTM model
    input_dim = n_components
    hidden_dim1 = 128
    hidden_dim2 = 256
    hidden_dim3 = 64
    output_dim = n_components
    
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3,
        output_dim=output_dim
    ).to(device)
    
    # 4. Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 5. Train model
    print("LSTM Training begins...")
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=200,
        patience=20
    )
    
    # Save the model
    torch.save(model.state_dict(), f"{lstm_dir}/best_model.pth")
    
    # 6. Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{data_config["name"]} - LSTM Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{lstm_dir}/training_history.png')
    plt.close()
    
    # 7. Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item() * X_batch.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Data set MSE: {test_loss:.6f}")
    
    # Compare with original data MSE
    # Get the corresponding original data for the test set
    test_start_idx = n_train + n_val
    test_end_idx = test_start_idx + len(X_test)
    original_data_test = x[test_start_idx:test_end_idx]

    # Inverse transform the predictions to original space
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            y_pred = y_pred.cpu().numpy()
            # Inverse PCA transform
            y_pred = pca.inverse_transform(y_pred)
            # Inverse scaling transform
            y_pred = scaler.inverse_transform(y_pred)
            all_preds.append(y_pred)

    all_preds = np.concatenate(all_preds, axis=0)
    all_preds_reshaped = all_preds.reshape(all_preds.shape[0], n_latitudes, n_longitudes)

    # Ensure the number of samples matches
    min_samples = min(original_data_test.shape[0], all_preds_reshaped.shape[0])
    original_data_test = original_data_test[:min_samples]
    all_preds_reshaped = all_preds_reshaped[:min_samples]

    # Calculate MSE between original data and predictions
    mse_with_original = np.mean((original_data_test - all_preds_reshaped) ** 2)
    print(f"MSE with Original Data: {mse_with_original:.6f}")
    
    # 8. Visualize predictions vs ground truth
    # Get the last sequence from test set
    final_sequence = X_pca_fft[int(-1-seq_length):-1]  # Last 100 steps (excluding the very last step)
    true_final = X_pca_fft[-1]  # The very last step

    # Make prediction using the model
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_sequence = torch.FloatTensor(final_sequence).unsqueeze(0).to(device)
        # Predict
        pred_final = model(input_sequence).cpu().numpy()[0]
    
    # Convert back to original space for visualization
    # true_next_orig = pca.inverse_transform(true_next.numpy().reshape(1, -1))
    # true_next_orig = scaler.inverse_transform(true_next_orig)
    # true_next_orig = true_next_orig.reshape(n_latitudes, n_longitudes)
    
    pred_final_orig = pca.inverse_transform(pred_final.reshape(1, -1))
    pred_final_orig = scaler.inverse_transform(pred_final_orig)
    pred_final_orig = pred_final_orig.reshape(n_latitudes, n_longitudes)
    
    # Compare with original data
    # plt.figure(figsize=(18, 6))

    # # Original data
    # plt.subplot(1, 3, 1)
    # plt.pcolormesh(lon, lat, x[-1, :, :], shading='auto', cmap='viridis')
    # plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    # plt.title('Original Data')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')

    # # Ground truth (PCA reconstructed)
    # plt.subplot(1, 3, 2)
    # plt.pcolormesh(lon, lat, pred_final_orig, shading='auto', cmap='viridis')
    # plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    # plt.title('True Next State (PCA Reconstructed)')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')

    plt.figure(figsize=(10, 6))
    # Error between original and reconstructed
    # plt.subplot(1, 3, 3)
    error_original = x[-1, :, :] - pred_final_orig
    print(f"Error between original and reconstructed: {np.mean(error_original**2):.6f}")
    plt.pcolormesh(lon, lat, error_original, cmap='RdBu_r', shading='auto')
    plt.colorbar(label=f'Error ({data_config["unit"]})')
    plt.title(f'Error Comparison')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(f'{lstm_dir}/lstm_error_comparison.png')
    plt.close()

    print(f"{data_config['name']} LSTM done, result in {lstm_dir} directory")

print("\nAll processes completed.")