"""
Análise de Filtros Digitais para Sinais Sísmicos
Autores: [Nomes dos integrantes]
Data: [Data]

Script principal para análise de filtros em sinais sísmicos.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos das funções
from src.sinal_sintetico import generate_synthetic_seismic_signal
from src.filtro_fft import butter_bandpass, butter_bandpass_filter, design_fir_bandpass_filter, apply_fir_filter
from src.analise_filtro import analyze_impulse_response, plot_impulse_response, plot_pole_zero_diagram
from src.calculo_metricas import calculate_metrics, print_metrics_table
from src.visualizacao import plot_time_domain, plot_frequency_domain, plot_filter_response


def main():
    """
    Função principal do programa.
    """
    print("="*60)
    print("ANÁLISE DE FILTROS DIGITAIS PARA SINAIS SÍSMICOS")
    print("="*60)
    
    # Configurações
    fs = 100.0  # Hz
    lowcut = 0.05  # Hz
    highcut = 1.0  # Hz
    order = 4
    
    print(f"\nConfigurações:")
    print(f"  Frequência de amostragem: {fs} Hz")
    print(f"  Banda de interesse: {lowcut}-{highcut} Hz")
    print(f"  Ordem do filtro: {order}")
    
    # 1. GERAR SINAL SINTÉTICO
    print("\n" + "-"*40)
    print("1. GERANDO SINAL SINTÉTICO")
    print("-"*40)
    
    duration = 300  # 5 minutos
    synthetic, clean, t, event_params = generate_synthetic_seismic_signal(fs, duration)
    
    print(f"  Duração: {duration} s ({len(synthetic)} amostras)")
    print(f"  Eventos simulados: {len(event_params)}")
    
    # 2. CARREGAR SINAL REAL (se disponível)
    print("\n" + "-"*40)
    print("2. CARREGANDO SINAL REAL")
    print("-"*40)
    
    try:
        from obspy import read
        st = read("data/terremoto_real.mseed")
        tr = st[0]
        real_data = tr.data
        real_times = tr.times()
        real_fs = tr.stats.sampling_rate
        
        print(f"  Sinal real carregado com sucesso!")
        print(f"  Pontos: {len(real_data)} | Fs: {real_fs} Hz")
        
        # Redimensionar se necessário
        if real_fs != fs:
            print(f"  Aviso: Fs do sinal real ({real_fs} Hz) diferente da configurada ({fs} Hz)")
    
    except Exception as e:
        print(f"  Erro ao carregar sinal real: {e}")
        print("  Usando apenas sinal sintético para análise.")
        real_data = None
    
    # 3. PROJETAR FILTROS
    print("\n" + "-"*40)
    print("3. PROJETANDO FILTROS")
    print("-"*40)
    
    # Filtro IIR Butterworth
    b_iir, a_iir = butter_bandpass(lowcut, highcut, fs, order)
    print(f"  Filtro IIR Butterworth (ordem {order}) projetado")
    print(f"    Coeficientes b: {len(b_iir)}, a: {len(a_iir)}")
    
    # Filtro FIR
    fir_taps = design_fir_bandpass_filter(lowcut, highcut, fs, numtaps=101, method='window')
    print(f"  Filtro FIR (101 taps, janela Hamming) projetado")
    
    # 4. APLICAR FILTROS AO SINAL SINTÉTICO
    print("\n" + "-"*40)
    print("4. APLICANDO FILTROS AO SINAL SINTÉTICO")
    print("-"*40)
    
    # Aplicar filtro IIR
    synthetic_iir = butter_bandpass_filter(synthetic, lowcut, highcut, fs, order)
    
    # Aplicar filtro FIR
    synthetic_fir = apply_fir_filter(synthetic, fir_taps, compensate_delay=True)
    
    print("  Filtros aplicados com sucesso!")
    
    # 5. CALCULAR MÉTRICAS
    print("\n" + "-"*40)
    print("5. CALCULANDO MÉTRICAS DE DESEMPENHO")
    print("-"*40)
    
    # Métricas para IIR
    metrics_iir = calculate_metrics(synthetic, synthetic_iir, clean, fs, lowcut, highcut)
    print_metrics_table(metrics_iir, "IIR Butterworth")
    
    # Métricas para FIR
    metrics_fir = calculate_metrics(synthetic, synthetic_fir, clean, fs, lowcut, highcut)
    print_metrics_table(metrics_fir, "FIR (Hamming)")
    
    # 6. VISUALIZAÇÕES
    print("\n" + "-"*40)
    print("6. GERANDO VISUALIZAÇÕES")
    print("-"*40)
    
    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 6.1 Sinal sintético no tempo
    axes[0, 0].plot(t, synthetic, 'gray', alpha=0.7, label='Sintético (com ruído)')
    axes[0, 0].plot(t, clean, 'r', linewidth=1, label='Limpo (eventos)', alpha=0.7)
    axes[0, 0].set_title("Sinal Sintético - Original")
    axes[0, 0].set_xlabel("Tempo (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Destacar eventos
    for event_name, params in event_params.items():
        axes[0, 0].axvspan(params['time'], params['time'] + params['duration'], 
                          alpha=0.2, color='yellow')
    
    # 6.2 Sinal filtrado IIR
    axes[0, 1].plot(t, synthetic_iir, 'b', label='Filtrado (IIR)')
    axes[0, 1].plot(t, clean, 'r', linewidth=1, label='Limpo', alpha=0.5)
    axes[0, 1].set_title("Sinal Filtrado - IIR Butterworth")
    axes[0, 1].set_xlabel("Tempo (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 6.3 Sinal filtrado FIR
    axes[1, 0].plot(t, synthetic_fir, 'g', label='Filtrado (FIR)')
    axes[1, 0].plot(t, clean, 'r', linewidth=1, label='Limpo', alpha=0.5)
    axes[1, 0].set_title("Sinal Filtrado - FIR Hamming")
    axes[1, 0].set_xlabel("Tempo (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 6.4 Espectro de frequência - Original
    freqs = np.fft.rfftfreq(len(synthetic), d=1/fs)
    fft_original = np.abs(np.fft.rfft(synthetic))
    fft_iir = np.abs(np.fft.rfft(synthetic_iir))
    fft_fir = np.abs(np.fft.rfft(synthetic_fir))
    
    axes[1, 1].semilogy(freqs, fft_original, 'gray', alpha=0.5, label='Original')
    axes[1, 1].semilogy(freqs, fft_iir, 'b', alpha=0.7, label='IIR')
    axes[1, 1].semilogy(freqs, fft_fir, 'g', alpha=0.7, label='FIR')
    axes[1, 1].axvspan(lowcut, highcut, alpha=0.2, color='yellow', label='Banda de interesse')
    axes[1, 1].set_title("Espectro de Frequência")
    axes[1, 1].set_xlabel("Frequência (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_xlim(0, 5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6.5 Resposta em frequência dos filtros
    from scipy.signal import freqz
    w_iir, h_iir = freqz(b_iir, a_iir, worN=2000)
    freq_iir = 0.5 * fs * w_iir / np.pi
    
    w_fir, h_fir = freqz(fir_taps, 1.0, worN=2000)
    freq_fir = 0.5 * fs * w_fir / np.pi
    
    axes[2, 0].plot(freq_iir, 20 * np.log10(np.abs(h_iir)), 'b', label='IIR Butterworth')
    axes[2, 0].plot(freq_fir, 20 * np.log10(np.abs(h_fir)), 'g', label='FIR Hamming')
    axes[2, 0].set_title("Resposta em Frequência dos Filtros")
    axes[2, 0].set_xlabel("Frequência (Hz)")
    axes[2, 0].set_ylabel("Ganho (dB)")
    axes[2, 0].set_xlim(0, 2)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6.6 Diagrama de polos e zeros (apenas IIR)
    from scipy.signal import tf2zpk
    z_iir, p_iir, k_iir = tf2zpk(b_iir, a_iir)
    
    # Círculo unitário
    theta = np.linspace(0, 2*np.pi, 100)
    axes[2, 1].plot(np.cos(theta), np.sin(theta), '--', color='gray', alpha=0.5)
    
    # Polos e zeros
    axes[2, 1].scatter(np.real(z_iir), np.imag(z_iir), s=50, marker='o',
                      facecolors='none', edgecolors='b', label='Zeros')
    axes[2, 1].scatter(np.real(p_iir), np.imag(p_iir), s=50, marker='x',
                      color='r', label='Polos')
    
    # Verificação de estabilidade
    if np.all(np.abs(p_iir) <= 1.0 + 1e-6):
        status = "ESTÁVEL"
        color = 'green'
    else:
        status = "INSTÁVEL"
        color = 'red'
    
    axes[2, 1].text(0, 0, status, fontsize=12, fontweight='bold',
                   color=color, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.7))
    
    axes[2, 1].set_title("Diagrama de Polos e Zeros (IIR)")
    axes[2, 1].set_xlabel("Parte Real")
    axes[2, 1].set_ylabel("Parte Imaginária")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axis('equal')
    axes[2, 1].set_xlim([-1.5, 1.5])
    axes[2, 1].set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('resultados_analise.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. ANÁLISE DE RESPOSTA IMPULSIVA
    print("\n" + "-"*40)
    print("7. ANÁLISE DE RESPOSTA IMPULSIVA")
    print("-"*40)
    
    response_iir, metrics_imp_iir = analyze_impulse_response(b_iir, a_iir, fs, "IIR Butterworth")
    print(f"  IIR Butterworth:")
    print(f"    Amplitude máxima: {metrics_imp_iir['max_amplitude']:.4f}")
    print(f"    Tempo de estabilização: {metrics_imp_iir['settling_time']:.3f} s")
    print(f"    Duração do ringing: {metrics_imp_iir['ringing_duration']:.3f} s")
    
    # Plot da resposta impulsiva
    fig_imp = plot_impulse_response(response_iir, fs, "IIR Butterworth")
    plt.show()
    
    # 8. RESUMO FINAL
    print("\n" + "="*60)
    print("RESUMO DA ANÁLISE")
    print("="*60)
    
    print(f"\nComparação de Desempenho:")
    print(f"{'Métrica':<25} {'IIR':<10} {'FIR':<10}")
    print("-"*45)
    
    for key in ['SNR_improvement_dB', 'RMSE_reduction_%', 'Correlation_filtered', 
                'Energy_ratio_improvement', 'Peaks_detected_filtered']:
        val_iir = metrics_iir.get(key, 'N/A')
        val_fir = metrics_fir.get(key, 'N/A')
        
        if isinstance(val_iir, float):
            val_iir_str = f"{val_iir:.2f}"
        else:
            val_iir_str = str(val_iir)
            
        if isinstance(val_fir, float):
            val_fir_str = f"{val_fir:.2f}"
        else:
            val_fir_str = str(val_fir)
        
        print(f"{key:<25} {val_iir_str:<10} {val_fir_str:<10}")
    
    print("\n" + "="*60)
    print("Análise concluída com sucesso!")
    print(f"Figura salva como: resultados_analise.png")
    print("="*60)


if __name__ == "__main__":
    main()