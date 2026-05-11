# 🍎 Redes Neurais Artificiais: Classificação de Frutas com MLP

Este repositório contém a implementação completa de uma **Rede Neural Artificial** do tipo **Perceptron de Múltiplas Camadas (MLP)**, construída **do zero (from scratch)** em Python puro. 

O projeto foi desenvolvido como trabalho prático para a disciplina de Inteligência Artificial da UFERSA, com o objetivo de demonstrar o funcionamento interno do algoritmo **Backpropagation** sem o uso de bibliotecas de machine learning prontas, como NumPy, TensorFlow, scikit-learn ou PyTorch.

---

## 📖 Sobre o Problema

O objetivo da rede neural é **classificar a qualidade de frutas** com base em duas características (como peso, tamanho ou espessura da borda), representadas por valores numéricos normalizados. A saída deve indicar se a fruta possui uma qualidade boa/excelente (`1.0`) ou ruim/média (`0.0`).

### Conjunto de Dados
A rede é alimentada por 20 amostras. Cada amostra contém 2 entradas (características) e 1 saída esperada (classe):
- **Entradas:** `[x1, x2]` (ex: `0.20, 0.15` indica Fruta pequena e leve)
- **Saída Desejada:** `0.0` (Ruim/Média) ou `1.0` (Boa/Excelente)

Como o padrão dos dados não é linearmente separável de forma trivial, foi obrigatória a implementação de pelo menos uma camada oculta.

---

## 🧠 Arquitetura e Implementação

Para fins acadêmicos e para a demonstração explícita da matemática da Descida do Gradiente (Gradient Descent), o projeto possui as seguintes características:

- **Linguagem:** Python Puro (apenas as bibliotecas nativas `math` e `random`).
- **Arquitetura Fixa:**
  - **Camada de Entrada:** 2 neurônios
  - **Camada Oculta:** 4 neurônios
  - **Camada de Saída:** 1 neurônio
- **Função de Ativação:** Sigmoide (implementada nativamente, incluindo sua derivada).
- **Algoritmo de Aprendizagem (Backpropagation):** Implementado passo a passo, compreendendo:
  1. Forward Pass (Propagação)
  2. Cálculo do Erro Local
  3. Retropropagação do erro utilizando a derivada da função Sigmoide
  4. Atualização dos pesos e biases das sinapses
- **Função de Custo:** Erro Médio Quadrático (MSE).

---

## 🚀 Como Executar

Por se tratar de um script Python puro, não há necessidade de criar ambientes virtuais complexos ou instalar dependências pelo `pip`.

1. Clone este repositório para a sua máquina local:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Execute o script principal:
   ```bash
   python3 mlp.py
   ```

**O que vai acontecer?**
O script imprimirá no terminal logs altamente didáticos. Ele mostrará:
1. O setup da arquitetura da rede.
2. A evolução da queda do **Erro Médio Quadrático (MSE)** a cada 1.000 épocas.
3. O resultado final dos testes de Classificação (Forward Pass) exibindo o valor esperado, o valor predito (probabilidade gerada pela rede) e a decisão final.

---

## 📈 Resultados e Desempenho

Com uma **Taxa de Aprendizado (Learning Rate)** configurada em `0.2` e um treinamento de **10.000 épocas**, a rede converge perfeitamente para este problema.

O Erro Médio Quadrático (MSE) cai de aproximadamente `0.26` na Época 1 para cerca de `0.0005` na Época 10.000. 

**Acurácia Final:** `100.00%` (acertando as 20 amostras do conjunto de treinamento).

---

