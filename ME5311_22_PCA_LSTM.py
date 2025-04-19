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

# PyTorch LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        # First LSTM layer - returns outputs for all time steps by default
        lstm1_out, _ = self.lstm1(x)  # lstm1_out shape: [batch, seq_len, hidden_dim1]
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # lstm2_out shape: [batch, seq_len, hidden_dim2]
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])  # Take only the last time step output

        # Output layer
        output = self.fc(lstm2_out)
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


# Visualization functions of PCA results
def plot_pca_explained_variance(explained_variance_ratio, n_components, data_config):
    """Plot the cumulative explained variance by number of components."""
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'{data_config["name"]} - Explained Variance by Components')
    plt.grid(True)
    plt.savefig(f'{data_config["dir"]}/pca_explained_variance.png')
    plt.close()
    
    return cumulative_variance_ratio

def plot_singular_value_spectrum(singular_values, n_components, data_config):
    """Plot the singular value spectrum in log scale."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, n_components + 1), singular_values, 'ro-')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title(f'{data_config["name"]} - Singular Value Spectrum')
    plt.grid(True)
    plt.savefig(f'{data_config["dir"]}/singular_value_spectrum.png')
    plt.close()

def plot_original_vs_reconstructed(x, X_reconstructed, time, lon, lat, time_idx, data_config):
    """Plot original data side by side with reconstructed data."""
    plt.figure(figsize=(18, 6))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.pcolormesh(lon, lat, x[time_idx], shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title(f'Original Data - {time[time_idx]}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Reconstructed data
    plt.subplot(1, 2, 2)
    plt.pcolormesh(lon, lat, X_reconstructed[time_idx], shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title(f'Reconstructed Data - {time[time_idx]}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/original_vs_reconstructed.png')
    plt.close()

def plot_reconstruction_error(x, X_reconstructed, lon, lat, time_idx, data_config):
    """Plot the reconstruction error map."""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon, lat, x[time_idx] - X_reconstructed[time_idx], 
                  cmap='RdBu_r', shading='auto')
    plt.colorbar(label=f'Error ({data_config["unit"]})')
    plt.title(f'{data_config["name"]} - Reconstruction Error')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/reconstruction_error.png')
    plt.close()

def plot_pca_modes_and_coefficients(pca, X_pca, time, lon, lat, n_latitudes, n_longitudes, data_config, n_modes_to_plot=6):
    # Plot spatial modes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(n_modes_to_plot):
        # Get the spatial mode for the i-th principal component
        mode = pca.components_[i].reshape(n_latitudes, n_longitudes)
        
        # Plot on the corresponding subplot
        im = axes[i].pcolormesh(lon, lat, mode, cmap='RdBu_r', shading='auto')
        axes[i].set_title(f'Spatial Mode {i+1}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/spatial_modes.png')
    plt.close()
    
    # Plot temporal coefficients
    plt.figure(figsize=(15, 10))
    for i in range(n_modes_to_plot):
        plt.subplot(n_modes_to_plot, 1, i+1)
        plt.plot(time, X_pca[:, i])
        plt.title(f'Temporal Coefficient of Mode {i+1}')
        plt.ylabel('Amplitude')
        if i == n_modes_to_plot-1:
            plt.xlabel('Time')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/temporal_coefficients.png')
    plt.close()

# Main visualization function that calls all the others
def visualize_pca_results(x, X_pca, pca, scaler, time, lon, lat, 
                         n_samples, n_latitudes, n_longitudes, data_config):
    """Generate all PCA visualization plots and print summary statistics."""
    # Get PCA components and explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    n_components = X_pca.shape[1]
    
    # Plot explained variance
    cumulative_variance_ratio = plot_pca_explained_variance(
        explained_variance_ratio, n_components, data_config)
    
    # Plot singular values
    plot_singular_value_spectrum(pca.singular_values_, n_components, data_config)
    
    # Reconstruct data
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    X_reconstructed = X_reconstructed.reshape(n_samples, n_latitudes, n_longitudes)
    
    # Calculate reconstruction error
    mse = np.mean((x - X_reconstructed) ** 2)
    print(f"{data_config['name']} mse of reconstruction: {mse}")
    
    # Use the last time point for visualization
    time_idx = -1  # Last snapshot
    
    # Plot original vs reconstructed
    plot_original_vs_reconstructed(
        x, X_reconstructed, time, lon, lat, time_idx, data_config)
    
    # Plot reconstruction error
    plot_reconstruction_error(
        x, X_reconstructed, lon, lat, time_idx, data_config)
    
    # Plot PCA modes and their temporal coefficients
    plot_pca_modes_and_coefficients(
        pca, X_pca, time, lon, lat, n_latitudes, n_longitudes, data_config)

    # Print summary statistics
    print(f"{data_config['name']} PCA dimensionality reduction: from {n_latitudes * n_longitudes} dimensions to {n_components} dimensions")
    print(f"{data_config['name']} Variance ratio explained by {n_components} principal components: {cumulative_variance_ratio[-1]:.4f}")
    
    # Plot component comparison
    # component_numbers = [5, 10, 20, 30, 50]
    # plot_component_comparison(
    #     x, X_pca, pca, scaler, explained_variance_ratio,
    #     lon, lat, n_samples, n_latitudes, n_longitudes,
    #     time_idx, data_config, component_numbers)
    
    return X_reconstructed, mse


# dimensions of data
n_samples = 16071
n_latitudes = 101 
n_longitudes = 161
shape = (n_samples, n_latitudes, n_longitudes)

# Data types to process
data_types = [
    {'file': 'data/slp.nc', 'var': 'msl', 'name': 'Sea Level Pressure', 'unit': 'Pa', 
     'dir': 'results_slp', 'lstm_dir': 'results_slp_lstm', 'n_components': 50},
    {'file': 'data/t2m.nc', 'var': 't2m', 'name': '2-meter Temperature', 'unit': 'K', 
     'dir': 'results_t2m', 'lstm_dir': 'results_t2m_lstm', 'n_components': 100}
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
    
    # After performing PCA
    X_reconstructed, mse = visualize_pca_results(
        x, X_pca, pca, scaler, time, lon, lat,
        n_samples, n_latitudes, n_longitudes, data_config
    )

    # =============== LSTM时间序列预测 ===============
    print(f"\n run {data_config['name']} 's LSTM time series prediction...")
    
    # Create LSTM results directory
    lstm_dir = data_config['lstm_dir']
    os.makedirs(lstm_dir, exist_ok=True)
    
    # 1. Split data into train/validation/test sets (8:1:1)
    n_total = X_pca.shape[0]
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    X_train = X_pca[:n_train]
    X_val = X_pca[n_train:n_train+n_val]
    X_test = X_pca[n_train+n_val:]
    
    print(f"Data split: Train {X_train.shape[0]} samples, Validation {X_val.shape[0]} samples, Test {X_test.shape[0]} samples")
    
    # 2. Create PyTorch datasets and dataloaders
    seq_length = 30  # Use past 30 days to predict next day
    batch_size = 32
    
    train_dataset = TimeSeriesDataset(X_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 3. Create LSTM model
    input_dim = n_components
    hidden_dim1 = 128
    hidden_dim2 = 64
    output_dim = n_components
    
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
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
        epochs=100,
        patience=10
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
    
    # 8. Visualize predictions vs ground truth
    # Get the last sequence from test set
    last_seq_idx = len(test_dataset) - 1
    last_sequence, true_next = test_dataset[last_seq_idx]
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        pred_next = model(last_sequence.unsqueeze(0).to(device)).cpu().numpy()[0]
    
    # Convert back to original space for visualization
    true_next_orig = pca.inverse_transform(true_next.numpy().reshape(1, -1))
    true_next_orig = scaler.inverse_transform(true_next_orig)
    true_next_orig = true_next_orig.reshape(n_latitudes, n_longitudes)
    
    pred_next_orig = pca.inverse_transform(pred_next.reshape(1, -1))
    pred_next_orig = scaler.inverse_transform(pred_next_orig)
    pred_next_orig = pred_next_orig.reshape(n_latitudes, n_longitudes)
    
    # Visualization
    plt.figure(figsize=(18, 6))
    
    # Ground truth
    plt.subplot(1, 3, 1)
    plt.pcolormesh(lon, lat, true_next_orig, shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title('True Next State')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolormesh(lon, lat, pred_next_orig, shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title('Predicted Next State')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Error
    plt.subplot(1, 3, 3)
    error = true_next_orig - pred_next_orig
    plt.pcolormesh(lon, lat, error, cmap='RdBu_r', shading='auto')
    plt.colorbar(label=f'Error ({data_config["unit"]})')
    plt.title(f'Prediction Error (MSE: {np.mean(error**2):.2f})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'{lstm_dir}/prediction_comparison.png')
    plt.close()
    
    # 9. Generate multi-step predictions
    multi_step_horizon = 10
    initial_sequence = last_sequence.numpy()
    
    multi_step_preds = generate_multi_step_predictions(
        model=model,
        initial_sequence=initial_sequence,
        steps=multi_step_horizon,
        device=device
    )
    
    # Convert predictions to original space
    multi_step_preds_orig = []
    for pred in multi_step_preds:
        pred_orig = pca.inverse_transform(pred.reshape(1, -1))
        pred_orig = scaler.inverse_transform(pred_orig)
        pred_orig = pred_orig.reshape(n_latitudes, n_longitudes)
        multi_step_preds_orig.append(pred_orig)
    
    # Visualize multi-step predictions
    steps_to_show = [0, 4, 9]  # Day 1, 5, 10
    
    plt.figure(figsize=(18, 6))
    for i, step in enumerate(steps_to_show):
        plt.subplot(1, 3, i+1)
        plt.pcolormesh(lon, lat, multi_step_preds_orig[step], shading='auto', cmap='viridis')
        plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
        plt.title(f'Prediction: t+{step+1} day')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'{lstm_dir}/multi_step_prediction.png')
    plt.close()
    
    print(f"{data_config['name']} LSTM done, result in {lstm_dir} directory")

print("\nAll processes completed.")