"""
Módulo para projeto e aplicação de filtros digitais.
"""

import numpy as np
from scipy.signal import butter, lfilter, freqz, group_delay, firwin, remez, kaiserord, firwin2, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Projeta filtro IIR Butterworth passa-faixa.
    
    Parâmetros:
    - lowcut, highcut: frequências de corte (Hz)
    - fs: frequência de amostragem (Hz)
    - order: ordem do filtro
    
    Retorna:
    - b, a: coeficientes do filtro
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Aplica filtro Butterworth passa-faixa.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def design_fir_bandpass_filter(lowcut, highcut, fs, numtaps=101, window='hamming', method='window'):
    """
    Projeta filtro FIR passa-faixa usando diferentes métodos.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if method == 'window':
        taps = firwin(numtaps, [low, high], pass_zero=False, window=window)
    
    elif method == 'remez':
        bands = [0, low*0.9, low, high, high*1.1, 1]
        desired = [0, 0, 1, 1, 0, 0]
        weight = [1, 10, 1]
        taps = remez(numtaps, bands, desired, weight=weight)
    
    elif method == 'firwin2':
        freq = [0, low*0.8, low, high, high*1.2, 1]
        gain = [0, 0, 1, 1, 0, 0]
        taps = firwin2(numtaps, freq, gain)
    
    elif method == 'kaiser':
        width = 0.1 * (high - low)
        ripple_db = 40.0
        N, beta = kaiserord(ripple_db, width)
        taps = firwin(N, [low, high], pass_zero=False, window=('kaiser', beta))
    
    else:
        raise ValueError(f"Método '{method}' não reconhecido. Use: 'window', 'remez', 'firwin2', 'kaiser'")
    
    return taps


def apply_fir_filter(data, taps, compensate_delay=False):
    """
    Aplica filtro FIR com opção de compensação de atraso.
    """
    if compensate_delay:
        filtered = filtfilt(taps, 1.0, data)
    else:
        filtered = lfilter(taps, 1.0, data)
    
    return filtered