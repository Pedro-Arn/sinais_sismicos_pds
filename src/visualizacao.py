"""
Módulo para visualização de sinais e resultados.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(times, original, filtered=None, title="Domínio do Tempo", 
                    xlabel="Tempo", ylabel="Amplitude", time_unit='s'):
    """
    Plota sinais no domínio do tempo.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if time_unit == 'min':
        times_plot = times / 60
        xlabel = "Tempo (minutos)"
    else:
        times_plot = times
        xlabel = f"Tempo ({time_unit})"
    
    ax.plot(times_plot, original, 'gray', alpha=0.5, label='Sinal Original')
    if filtered is not None:
        ax.plot(times_plot, filtered, 'r', linewidth=1.5, label='Sinal Filtrado')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_frequency_domain(fs, original, filtered=None, title="Domínio da Frequência", 
                         xlabel="Frequência (Hz)", ylabel="Magnitude", max_freq=None):
    """
    Plota espectro de frequência.
    """
    freqs = np.fft.rfftfreq(len(original), d=1/fs)
    fft_original = np.abs(np.fft.rfft(original))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.semilogy(freqs, fft_original, 'gray', alpha=0.5, label='Espectro Original')
    if filtered is not None:
        fft_filtered = np.abs(np.fft.rfft(filtered))
        ax.semilogy(freqs, fft_filtered, 'b', linewidth=1.5, label='Espectro Filtrado')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if max_freq is not None:
        ax.set_xlim(0, max_freq)
    
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_filter_response(b, a, fs, title="Resposta em Frequência do Filtro"):
    """
    Plota resposta em frequência do filtro.
    """
    from scipy.signal import freqz
    
    w, h = freqz(b, a, worN=2000)
    freqs = 0.5 * fs * w / np.pi
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(freqs, 20 * np.log10(np.abs(h)), 'g')
    ax.set_title(title)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Ganho (dB)")
    ax.grid(True)
    ax.set_xlim(0, 2)
    
    return fig