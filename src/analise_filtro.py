"""
Módulo para análise de filtros (resposta impulsiva, polos/zeros).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter


def analyze_impulse_response(b, a, fs, filter_name="Filtro"):
    """
    Analisa e plota a resposta ao impulso do filtro.
    """
    from scipy.signal import lfilter
    
    # Resposta ao impulso
    impulse = np.zeros(1000)
    impulse[0] = 1
    response = lfilter(b, a, impulse)
    
    # Métricas da resposta impulsiva
    energy_cumulative = np.cumsum(response**2)
    
    metrics = {
        'max_amplitude': np.max(np.abs(response)),
        'energy_total': np.sum(response**2),
        'settling_time': np.argmax(energy_cumulative > 0.99 * energy_cumulative[-1]) / fs,
        'ringing_duration': len(response[np.abs(response) > 0.01 * np.max(np.abs(response))]) / fs
    }
    
    return response, metrics


def plot_impulse_response(response, fs, filter_name="Filtro"):
    """
    Plota a resposta ao impulso.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Resposta ao impulso
    axes[0, 0].stem(np.arange(len(response)), response, basefmt=" ")
    axes[0, 0].set_title(f"{filter_name} - Resposta ao Impulso")
    axes[0, 0].set_xlabel("Amostras")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Resposta ao degrau
    step_response = np.cumsum(response)
    axes[0, 1].plot(step_response, 'g')
    axes[0, 1].set_title(f"{filter_name} - Resposta ao Degrau")
    axes[0, 1].set_xlabel("Amostras")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energia cumulativa
    energy_cumulative = np.cumsum(response**2)
    axes[1, 0].plot(energy_cumulative, 'm')
    axes[1, 0].set_title(f"{filter_name} - Energia Cumulativa")
    axes[1, 0].set_xlabel("Amostras")
    axes[1, 0].set_ylabel("Energia")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resposta ao impulso no tempo real
    time_vector = np.arange(len(response)) / fs
    axes[1, 1].plot(time_vector, response, 'b')
    axes[1, 1].set_title(f"{filter_name} - Resposta ao Impulso (Tempo Real)")
    axes[1, 1].set_xlabel("Tempo (s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pole_zero_diagram(b=None, a=None, z=None, p=None, k=None, fs=None, order=4):
    """
    Plota diagrama de polos e zeros.
    """
    if z is None or p is None or k is None:
        if b is not None and a is not None and fs is not None:
            # Extrai polos e zeros dos coeficientes
            from scipy.signal import tf2zpk
            z, p, k = tf2zpk(b, a)
        else:
            raise ValueError("Forneça (b, a, fs) ou (z, p, k)")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Círculo Unitário
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Círculo Unitário')
    
    # Polos e Zeros
    ax.scatter(np.real(z), np.imag(z), s=50, marker='o', 
               facecolors='none', edgecolors='b', label='Zeros')
    ax.scatter(np.real(p), np.imag(p), s=50, marker='x', color='r', label='Polos')
    
    # Verificação de estabilidade
    if np.all(np.abs(p) <= 1.0 + 1e-6):
        status_text = "SISTEMA ESTÁVEL\n(Polos dentro do Círculo)"
        box_color = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.5)
    else:
        status_text = "SISTEMA INSTÁVEL\n(Polo fora do Círculo)"
        box_color = dict(boxstyle="round,pad=0.5", fc="mistyrose", ec="red", alpha=0.5)
    
    ax.text(0, 0, status_text, fontsize=12, fontweight='bold',
           color='black', ha='center', va='center', bbox=box_color)
    
    ax.set_title(f"Diagrama de Polos e Zeros")
    ax.set_xlabel("Parte Real")
    ax.set_ylabel("Parte Imaginária")
    ax.grid(True)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    
    return fig