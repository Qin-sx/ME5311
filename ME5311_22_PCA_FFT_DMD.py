import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create results directories
os.makedirs('results_slp', exist_ok=True)
os.makedirs('results_t2m', exist_ok=True)


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


from pydmd import DMD

def perform_dmd(X_pca_fft):
    """Perform Dynamic Mode Decomposition on FFT-processed PCA components and predict next time step."""
    # The input of the DMD function needs to be two-dimensional, with shape (n_features, n_samples)
    # The shape of PCA data is (n_samples, n_components), so it needs to be transposed
    X_pca_fft_T = X_pca_fft[:-1, :].T  # Exclude the last time step data
    # print(f"X_pca_fft_T shape: {X_pca_fft_T.shape}")
    # Initialize DMD
    dmd = DMD(svd_rank=-1)  # Use all singular values
    dmd.fit(X_pca_fft_T)
    
    # Get DMD modes and dynamics
    dmd_modes = dmd.modes
    dmd_dynamics = dmd.dynamics
    dmd_eigs = dmd.eigs
    
    # Predict the dynamic coefficients of the next time step (i.e., the last time step of the original data)
    next_dynamics = np.zeros_like(dmd_dynamics[:, 0])
    for i in range(len(dmd_eigs)):
        next_dynamics[i] = dmd_dynamics[-1, i] * dmd_eigs[i]
    
    # Reconstruct the state of the next time step
    next_state = dmd_modes @ next_dynamics
    
    # Convert the state of the next time step back to the original sample shape
    next_state = next_state.real
    next_state = next_state.reshape(1, -1)

    return next_state

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
    
    # Apply FFT to PCA components
    threshold_percent = 2.0  # Set your threshold percentage here
    X_pca_fft = apply_fft(X_pca, threshold_percent)
    

    next_state = perform_dmd(X_pca_fft)
    # print(f"Next state shape: {next_state.shape}")
    next_state_reconstructed = pca.inverse_transform(next_state)
    next_state_reconstructed = scaler.inverse_transform(next_state_reconstructed)
    next_state_reconstructed = next_state_reconstructed.reshape(1, n_latitudes, n_longitudes)

    # calculate the MSE between the original last state and the predicted state
    original_last_state = x[-1]
    predicted_state = next_state_reconstructed[0]
    mse = np.mean((original_last_state - predicted_state) ** 2)
    print(f"{data_config['name']} prediction MSE: {mse}")

    # Plot the original last state, predicted state, and error comparison
    plt.figure(figsize=(15, 6))

# original data
plt.figure(figsize=(10, 6))
plt.pcolormesh(lon, lat, original_last_state, shading='auto', cmap='viridis')
plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
plt.title(f'Original Data - Last Time Step')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(f'{data_config["dir"]}/DMD_original_data.png')
plt.close()

# predicted data
plt.figure(figsize=(10, 6))
plt.pcolormesh(lon, lat, predicted_state, shading='auto', cmap='viridis')
plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
plt.title(f'Predicted Data - Next Time Step')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(f'{data_config["dir"]}/DMD_predicted_data.png')
plt.close()

# error comparison
plt.figure(figsize=(10, 6))
plt.pcolormesh(lon, lat, original_last_state - predicted_state, 
               cmap='RdBu_r', shading='auto', vmin=-1000, vmax=1000)  # 可以根据实际情况调整vmin和vmax的值
plt.colorbar(label=f'Error ({data_config["unit"]})')
plt.title(f'Error Comparison')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(f'{data_config["dir"]}/DMD_error_comparison.png')
plt.close()


print("\nAll processes completed.")