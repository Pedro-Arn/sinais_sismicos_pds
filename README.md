# 📊 Análise de Filtros Digitais para Sinais Sísmicos

Este projeto implementa a análise e aplicação de filtros digitais (IIR e FIR) para processamento de sinais sísmicos, com o objetivo de detectar eventos sísmicos em sinais ruidosos.

## 🎯 Objetivos

- Implementar filtros passa-faixa para isolar frequências de interesse (0.05-1.0 Hz)
- Analisar sinais via Transformada Z e FFT
- Avaliar resposta impulsiva e estabilidade dos filtros
- Comparar desempenho entre filtros IIR (Butterworth) e FIR

## 📁 Estrutura do Projeto

```
sinais_sismicos_pds/
├── dados/                        # Dados sísmicos
│   └── terremoto_real.mseed    # Dataset real (Tohoku 2011)
└── src/                         # Módulos fonte
    ├── analise_filtro.py      # Análise de filtros
    ├── calculo_metricas.py  # Métricas de desempenho
    ├── filtro_fft.py           # Projeto de filtros
    ├── sinal_sintetico.py      # Geração de sinais sintéticos
    └── visualizacao.py        # Funções de plotagem
├── analise_filtro_sismico.py    # Script principal
├── requirements.txt             # Dependências
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Pedro-Arn/sinais_sismicos_pds.git
cd sinais_sismicos_pds
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Coloque o dataset real na pasta `dados/`:
- Arquivo: `terremoto_real.mseed`
- Execute o código em: download_sinal_bruto.py

## 💻 Uso

Execute o script principal:
```bash
python analise_filtro_sismico.py
```

O script irá:
1. Gerar um sinal sísmico sintético com ruído
2. Aplicar filtros IIR Butterworth e FIR
3. Calcular métricas de desempenho (SNR, RMSE, etc.)
4. Gerar visualizações completas
5. Salvar os resultados em `resultados_analise.png`

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **NumPy**: Processamento numérico
- **SciPy**: Filtros digitais e processamento de sinais
- **Matplotlib**: Visualização de dados
- **ObsPy**: Leitura de dados sísmicos (formato MSEED)

## 📈 Funcionalidades

### ✅ Implementadas
- [x] Geração de sinais sísmicos sintéticos com eventos e ruído
- [x] Filtro IIR Butterworth passa-faixa
- [x] Filtro FIR com múltiplos métodos (janela, Parks-McClellan)
- [x] Análise espectral via FFT
- [x] Diagrama de polos e zeros (estabilidade)
- [x] Resposta impulsiva dos filtros
- [x] Métricas quantitativas (SNR, RMSE, correlação)

## 📊 Resultados Esperados

- Redução eficiente de ruído em sinais sísmicos
- Preservação de eventos sísmicos na banda de interesse
- Detecção automática de picos correspondentes a eventos
- Comparação quantitativa entre filtros IIR e FIR