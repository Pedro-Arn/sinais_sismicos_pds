import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, gausspulse
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_seismic_signal(fs, duration, event_params=None):
    """
    Gera um sinal sísmico sintético com eventos e ruídos realistas.
    
    Parâmetros:
    - fs: frequência de amostragem (Hz)
    - duration: duração em segundos
    - event_params: dicionário com parâmetros dos eventos
    
    Retorna:
    - sinal_sintetico: array com o sinal sintético
    - sinal_limpo: array com apenas os eventos (sem ruído)
    - t: vetor de tempo
    """
    
    if event_params is None:
        event_params = {
            'main_event': {'time': 60, 'duration': 10, 'freq_range': (0.1, 0.5)},
            'aftershock1': {'time': 120, 'duration': 5, 'freq_range': (0.2, 1.0)},
            'aftershock2': {'time': 180, 'duration': 3, 'freq_range': (0.5, 2.0)}
        }
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sinal_limpo = np.zeros_like(t)
    
    # 1. Evento sísmico principal (ondas P e S)
    for event_name, params in event_params.items():
        event_start = params['time']
        event_duration = params['duration']
        freq_start, freq_end = params['freq_range']
        
        # Cria uma janela temporal para o evento
        event_mask = (t >= event_start) & (t <= event_start + event_duration)
        event_t = t[event_mask] - event_start
        
        # Onda P (mais rápida, alta frequência)
        wave_p = gausspulse(event_t, fc=(freq_start + freq_end)/2, bw=0.3)
        
        # Onda S (mais lenta, baixa frequência) - deslocada no tempo
        wave_s = np.zeros_like(event_t)
        if len(event_t) > int(fs * 2):  # Atraso de 2 segundos para onda S
            s_start = int(fs * 2)
            wave_s[s_start:] = 0.7 * gausspulse(event_t[:-s_start], 
                                                fc=freq_start, bw=0.2)
        
        sinal_limpo[event_mask] += wave_p + wave_s
    
    # 2. Adiciona ruído realista
    # Ruído de alta frequência (microsismos)
    noise_hf = 0.1 * np.sin(2 * np.pi * 5 * t) + 0.05 * np.sin(2 * np.pi * 10 * t)
    
    # Ruído de baixa frequência (vento, maré)
    noise_lf = 0.05 * np.sin(2 * np.pi * 0.02 * t) + 0.03 * np.sin(2 * np.pi * 0.05 * t)
    
    # Ruído aleatório (branco)
    noise_white = 0.15 * np.random.randn(len(t))
    
    # Ruído impulsivo (passos, tráfego)
    noise_impulsive = np.zeros_like(t)
    for i in range(10):
        idx = np.random.randint(0, len(t))
        noise_impulsive[idx:idx+100] += 0.3 * np.random.randn(100)
    
    sinal_sintetico = sinal_limpo + noise_hf + noise_lf + noise_white + noise_impulsive
    
    return sinal_sintetico, sinal_limpo, t, event_params

# Uso no código principal:
fs = 100.0  # Hz
duration = 300  # 5 minutos
sinal_sintetico, sinal_limpo, t, event_params = generate_synthetic_seismic_signal(fs, duration)

# Plot comparativo
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sinal_sintetico, 'gray', alpha=0.7, label='Sinal Sintético (com ruído)')
plt.plot(t, sinal_limpo, 'r', linewidth=1.5, label='Sinal Limpo (eventos)')
plt.title("Sinal Sísmico Sintético - Eventos + Ruído")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Destacar eventos
for event_name, params in event_params.items():
    plt.axvspan(params['time'], params['time'] + params['duration'], 
                alpha=0.2, color='yellow')
    plt.text(params['time'] + params['duration']/2, 
             plt.ylim()[1]*0.9, event_name, ha='center')

plt.tight_layout()
plt.show()