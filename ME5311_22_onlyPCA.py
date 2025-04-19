import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg

# 创建结果目录
os.makedirs('results_slp', exist_ok=True)
os.makedirs('results_t2m', exist_ok=True)

# dimensions of data
n_samples = 16071
n_latitudes = 101 
n_longitudes = 161
shape = (n_samples, n_latitudes, n_longitudes)

# 处理两种数据类型
data_types = [
    {'file': 'data/slp.nc', 'var': 'msl', 'name': 'Sea Level Pressure', 'unit': 'Pa', 
     'dir': 'results_slp', 'dmd_dir': 'results_slp_dmd', 'n_components': 50},
    {'file': 'data/t2m.nc', 'var': 't2m', 'name': '2-meter Temperature', 'unit': 'K', 
     'dir': 'results_t2m', 'dmd_dir': 'results_t2m_dmd', 'n_components': 100}
]

# 对每种数据类型进行PCA分析
for data_config in data_types:
    print(f"\n处理 {data_config['name']} 数据...")
    
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
    
    # 计算解释方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制解释方差
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'{data_config["name"]} - Explained Variance by Components')
    plt.grid(True)
    plt.savefig(f'{data_config["dir"]}/pca_explained_variance.png')
    plt.close()
    
    # 可视化奇异值谱
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, n_components + 1), pca.singular_values_, 'ro-')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title(f'{data_config["name"]} - Singular Value Spectrum')
    plt.grid(True)
    plt.savefig(f'{data_config["dir"]}/singular_value_spectrum.png')
    plt.close()
    
    # # 可视化前六个主成分的空间模态
    # n_modes_to_plot = 6
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # axes = axes.flatten()
    
    # for i in range(n_modes_to_plot):
    #     # 获取第i个主成分对应的空间模态
    #     mode = pca.components_[i].reshape(n_latitudes, n_longitudes)
        
    #     # 在对应的子图上绘制
    #     im = axes[i].pcolormesh(lon, lat, mode, cmap='RdBu_r', shading='auto')
    #     axes[i].set_title(f'Spatial Mode {i+1}')
    #     axes[i].set_xlabel('Longitude')
    #     axes[i].set_ylabel('Latitude')
    #     fig.colorbar(im, ax=axes[i])
    
    # plt.tight_layout()
    # plt.savefig(f'{data_config["dir"]}/spatial_modes.png')
    # plt.close()
    
    # # 可视化前六个主成分的时间系数
    # plt.figure(figsize=(15, 10))
    # for i in range(n_modes_to_plot):
    #     plt.subplot(n_modes_to_plot, 1, i+1)
    #     plt.plot(time, X_pca[:, i])
    #     plt.title(f'Temporal Coefficient of Mode {i+1}')
    #     plt.ylabel('Amplitude')
    #     if i == n_modes_to_plot-1:
    #         plt.xlabel('Time')
    #     plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig(f'{data_config["dir"]}/temporal_coefficients.png')
    # plt.close()
    
    # 重建数据
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    X_reconstructed = X_reconstructed.reshape(n_samples, n_latitudes, n_longitudes)
    
    # 计算重建误差
    mse = np.mean((x - X_reconstructed) ** 2)
    print(f"{data_config['name']} 重建均方误差: {mse}")
    
    # 可视化原始数据与重建数据(最后一个时间点)
    time_idx = -1  # 最后一个快照 (2022-12-31)
    
    # 原始数据
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(lon, lat, x[time_idx], shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title(f'Original Data - {time[time_idx]}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # 重建数据
    plt.subplot(1, 2, 2)
    plt.pcolormesh(lon, lat, X_reconstructed[time_idx], shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title(f'Reconstructed Data - {time[time_idx]}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/original_vs_reconstructed.png')
    plt.close()
    
    # 可视化重建误差
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
    
    print(f"{data_config['name']} PCA降维: 从{n_latitudes * n_longitudes}维到{n_components}维")
    print(f"{data_config['name']} {n_components}个主成分解释的方差比例: {cumulative_variance_ratio[-1]:.4f}")
    
    # 额外比较不同数量主成分的效果
    component_numbers = [5, 10, 20, 30, 50]
    plt.figure(figsize=(20, 12))
    
    # 原始数据
    plt.subplot(2, 3, 1)
    plt.pcolormesh(lon, lat, x[time_idx], shading='auto', cmap='viridis')
    plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
    plt.title('Original Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # 不同主成分数量的重建
    for i, n_comp in enumerate(component_numbers):
        if n_comp <= n_components:
            # 创建只保留前n_comp个主成分的数据
            X_pca_reduced = np.zeros_like(X_pca)
            X_pca_reduced[:, :n_comp] = X_pca[:, :n_comp]

            # 使用完整的主成分矩阵进行重建
            X_recon_partial = pca.inverse_transform(X_pca_reduced)
            X_recon_partial = scaler.inverse_transform(X_recon_partial)
            X_recon_partial = X_recon_partial.reshape(n_samples, n_latitudes, n_longitudes)

            # 计算重建误差和解释方差
            mse_partial = np.mean((x - X_recon_partial) ** 2)
            var_explained = np.sum(explained_variance_ratio[:n_comp])
            
            # 绘制重建结果
            plt.subplot(2, 3, i+2)
            plt.pcolormesh(lon, lat, X_recon_partial[time_idx], shading='auto', cmap='viridis')
            plt.colorbar(label=f'{data_config["name"]} ({data_config["unit"]})')
            plt.title(f'{n_comp} Components\nMSE: {mse_partial:.2f}\nVar: {var_explained:.4f}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(f'{data_config["dir"]}/component_comparison.png')
    plt.close()

print("\n分析完成！所有结果已保存到对应目录")