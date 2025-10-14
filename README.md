# Analise de Dados - Sequência de Proteínas
  **Aluna:** Gabriela Pedroso dos Santos Pontes<br/>
  **Professor:** Aryel Marlus Repula de Oliveira<br/><br/>

  🗂️ **Banco de Dados Utilizado:** https://scop.berkeley.edu/astral/ver=2.08<br/><br/>

**Descrição:**<br/>
Realizar análise de dados utilizando as técnicas discutidas em sala sobre não-supervisionado utilizando clusters. Utilize as IAs generativas para auxiliar na criação do código.
<br/><br/>
**Passos:**<br/>
📌Extrair os atributos de sequência segundo o que fizemos em sala, um kmer de 2X2, onde 2 são os pares de aminoácidos, e X é um skip. Fazer uma janela móvel deste kmer que extraia ausência e presença de todos os elementos das sequências e a transforme em uma matrix binária com todas as combinações possíveis;<br/>
📌Projetar a matriz geradas em uma base de PCA utilizando os 300 componentes principais;<br/>
📌Realizar a escolha do algoritmo de cluster rodando todos os algoritmos do sci-kit no python, calculando as métricas internas e as correlacionando com a métrica externa F1-Score;<br/>
📌Variar os parâmetros para obter a melhor métrica interna e sugerir a melhor configuração de clusters;<br/>


