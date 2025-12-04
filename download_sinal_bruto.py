import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Criar a pasta data/ se não existir
pasta_projeto = os.path.dirname(os.path.abspath(__file__))
pasta_data = os.path.join(pasta_projeto, "dados")

if not os.path.exists(pasta_data):
    os.makedirs(pasta_data)
    print(f"Pasta 'data' criada em: {pasta_data}")

# Caminho completo para o arquivo
caminho_completo = os.path.join(pasta_data, "terremoto_real.mseed")
print(f"O arquivo será salvo em: {caminho_completo}")

# ------------------------------------------------
# Conectar ao servidor IRIS
client = Client("IRIS")

# Escolhido o terremoto (Tohoku, Japão - 2011)
t = UTCDateTime("2011-03-11T05:46:24")
starttime = t
endtime = t + 60 * 60  # 1 hora de dados

print("Baixando dados da estação ANMO...")

try:
    # Baixar os dados
    st = client.get_waveforms("IU", "ANMO", "*", "BHZ", starttime, endtime)

    # Pegar o primeiro sinal e limpar (remove tendência)
    tr = st[0]
    tr.detrend("demean")

    # SALVAR NA PASTA data/
    tr.write(caminho_completo, format="MSEED")

    print("-" * 30)
    print("SUCESSO!")
    print(f"Arquivo salvo em: {caminho_completo}")
    print(f"Tamanho do sinal: {len(tr.data)} pontos")
    print(f"Frequência de amostragem: {tr.stats.sampling_rate} Hz")
    print(f"Duração: {tr.stats.endtime - tr.stats.starttime} segundos")
    print("-" * 30)

    # Preview rápido
    tr.plot()

except Exception as e:
    print(f"Erro: {e}")
    print("Verifique sua conexão com a internet ou tente novamente.")