"""
Módulo para cálculo de métricas de desempenho.
"""

import numpy as np
from scipy.signal import find_peaks


def calculate_metrics(original, filtered, clean_signal=None, fs=1.0, lowcut=None, highcut=None):
    """
    Calcula métricas quantitativas de desempenho do filtro.
    """
    metrics = {}
    
    # 1. SNR (Signal-to-Noise Ratio) - se tiver sinal limpo
    if clean_signal is not None:
        noise_original = original - clean_signal
        noise_filtered = filtered - clean_signal
        
        signal_power = np.sum(clean_signal**2)
        noise_power_original = np.sum(noise_original**2)
        noise_power_filtered = np.sum(noise_filtered**2)
        
        if noise_power_original > 0:
            snr_original = 10 * np.log10(signal_power / noise_power_original)
            metrics['SNR_original_dB'] = snr_original
        
        if noise_power_filtered > 0:
            snr_filtered = 10 * np.log10(signal_power / noise_power_filtered)
            metrics['SNR_filtered_dB'] = snr_filtered
            metrics['SNR_improvement_dB'] = snr_filtered - snr_original
    
    # 2. RMSE (Root Mean Square Error)
    if clean_signal is not None:
        metrics['RMSE_original'] = np.sqrt(np.mean((original - clean_signal)**2))
        metrics['RMSE_filtered'] = np.sqrt(np.mean((filtered - clean_signal)**2))
        if metrics['RMSE_original'] > 0:
            metrics['RMSE_reduction_%'] = 100 * (1 - metrics['RMSE_filtered'] / metrics['RMSE_original'])
    
    # 3. Correlation Coefficient
    if clean_signal is not None:
        corr_coef_original = np.corrcoef(clean_signal, original)[0, 1]
        corr_coef_filtered = np.corrcoef(clean_signal, filtered)[0, 1]
        metrics['Correlation_original'] = corr_coef_original
        metrics['Correlation_filtered'] = corr_coef_filtered
    
    # 4. Energy Ratio (banda de interesse vs banda total)
    fft_original = np.abs(np.fft.rfft(original))
    fft_filtered = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(original), d=1/fs)
    
    if lowcut is not None and highcut is not None:
        freq_mask_interest = (freqs >= lowcut) & (freqs <= highcut)
        freq_mask_outside = ~freq_mask_interest
        
        energy_original_interest = np.sum(fft_original[freq_mask_interest]**2)
        energy_original_outside = np.sum(fft_original[freq_mask_outside]**2)
        energy_filtered_interest = np.sum(fft_filtered[freq_mask_interest]**2)
        energy_filtered_outside = np.sum(fft_filtered[freq_mask_outside]**2)
        
        metrics['Energy_ratio_original'] = energy_original_interest / (energy_original_outside + 1e-10)
        metrics['Energy_ratio_filtered'] = energy_filtered_interest / (energy_filtered_outside + 1e-10)
        metrics['Energy_ratio_improvement'] = metrics['Energy_ratio_filtered'] / metrics['Energy_ratio_original']
    
    # 5. Peak Detection Metrics
    peaks_original, _ = find_peaks(np.abs(original), height=np.std(original), distance=int(fs/10))
    peaks_filtered, _ = find_peaks(np.abs(filtered), height=np.std(filtered), distance=int(fs/10))
    
    metrics['Peaks_detected_original'] = len(peaks_original)
    metrics['Peaks_detected_filtered'] = len(peaks_filtered)
    
    # 6. Variance Reduction
    metrics['Variance_original'] = np.var(original)
    metrics['Variance_filtered'] = np.var(filtered)
    metrics['Variance_reduction_%'] = 100 * (1 - metrics['Variance_filtered'] / metrics['Variance_original'])
    
    return metrics


def print_metrics_table(metrics, filter_name="Filtro"):
    """
    Imprime as métricas em formato de tabela.
    """
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DE DESEMPENHO - {filter_name}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'dB' in key or 'SNR' in key:
                print(f"{key:30s}: {value:8.2f} dB")
            elif 'ratio' in key or 'Correlation' in key:
                print(f"{key:30s}: {value:8.4f}")
            elif '%' in key:
                print(f"{key:30s}: {value:8.1f} %")
            else:
                print(f"{key:30s}: {value:8.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    print(f"{'='*60}")