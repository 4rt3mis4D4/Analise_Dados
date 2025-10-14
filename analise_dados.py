from Bio import SeqIO # Manipular sequências biológicas (DNA, RNA, proteínas) - formato FASTA
import os # Interagir com o sistema operacional (Verifica se o arquivo existe)
import sys # Acessar parâmetros e funcionalidades do sistema Python (Encerrar o Script)
import pandas as pd # Manipulação e análise de dados em tabelas

def extrair(arquivo):
    # Extrai as sequências em formato texto de um arquivo FASTA
    sequencias = []
    with open(arquivo, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            sequencias.append(str(record.seq))
    return sequencias

def gerar_kmers(sequencia, k1=2, skip=1):
    # Gera kmers (pares de aminoácidos) com espaçamento definido por "skip"
    kmers = []
    for i in range(len(sequencia) - (k1 - 1) - skip):
        parte1 = sequencia[i]
        parte2 = sequencia[i + 1 + skip]
        kmers.append(parte1 + parte2)
    return kmers

# Lista de arquivos para processar
arquivos_para_leitura = [
    "sequencia_proteinas_40.txt",
    "sequencia_proteinas_95.txt"
]

sequencias_proteinas = []

# Leitura dos arquivos
for arquivo in arquivos_para_leitura:
    if not os.path.exists(arquivo):
        print(f"ERRO: O arquivo '{arquivo}' não foi encontrado. Encerrando o programa.")
        sys.exit(1)

    print(f"Processando o arquivo: {arquivo}...")
    sequencias_proteinas.extend(extrair(arquivo))

print(f"\nTotal de sequências encontradas: {len(sequencias_proteinas)}")
print("\nPrimeiras 5 sequências e seus kmers 2x2 com skip:\n")

# Exibe exemplos de kmers
for i, seq in enumerate(sequencias_proteinas[:5], 1):
    kmers = gerar_kmers(seq)
    print(f"{i}. Sequência: {seq}")
    print(f"{kmers}\n")

# Coleta de todos os kmers únicos
todos_kmers = set()
for seq in sequencias_proteinas:
    kmers_seq = gerar_kmers(seq)
    todos_kmers.update(kmers_seq)

# Ordena para garantir consistência das colunas
todos_kmers = sorted(list(todos_kmers))

# Montagem da matriz binária
matriz_binaria = []

for seq in sequencias_proteinas:
    kmers_seq = gerar_kmers(seq)
    linha = [1 if kmer in kmers_seq else 0 for kmer in todos_kmers]
    matriz_binaria.append(linha)

# Cria DataFrame para visualização
df_matriz = pd.DataFrame(matriz_binaria, columns=todos_kmers)

print("\nMatriz binária de presença/ausência de kmers (exibindo as 5 primeiras linhas):")
print(df_matriz.head())

# Salvar a matriz em CSV (opcional)
df_matriz.to_csv("matriz_binaria_kmers.csv", index=False)
print("\nMatriz salva em 'matriz_binaria_kmers.csv'")
