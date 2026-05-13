# MLP - Classificação de Frutas (Backpropagation do Zero)

Este projeto consiste na implementação de uma Rede Neural Artificial do tipo **Multilayer Perceptron (MLP)**, desenvolvida integralmente em Python sem o auxílio de bibliotecas externas (como NumPy ou TensorFlow). O objetivo é classificar frutas com base em duas características de entrada (ex: peso e tamanho) em duas categorias possíveis (saída binária).

---

## 1. Arquitetura da Rede

A rede utiliza uma topologia **2-4-1**:

* 
**Camada de Entrada:** 2 neurônios (correspondentes às características das frutas).


* 
**Camada Oculta:** 4 neurônios com função de ativação não-linear (Sigmoide).


* **Camada de Saída:** 1 neurônio (classificação entre 0.0 e 1.0).

---

## 2. Fundamentação Matemática

### Função de Ativação (Sigmoide)

Para que a rede aprenda padrões não-lineares, utilizamos a função logística:


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Sua derivada, necessária para o ajuste de pesos, é:


$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

### Algoritmo Backpropagation

O aprendizado ocorre através do ajuste de pesos baseado no erro:

1. **Forward Pass:** As entradas são propagadas até a saída.
2. 
**Cálculo do Erro:** Diferença entre o valor esperado e o obtido.


3. **Retropropagação:** O erro é enviado de volta da saída para a entrada, calculando o gradiente local para cada peso.
4. **Atualização:** Os pesos são ajustados usando a Taxa de Aprendizado ($\eta$):

$$w_{novo} = w_{antigo} + (\eta \cdot \delta \cdot entrada)$$



---

## 3. Estrutura do Código (`mlp.py`)

* **`__init__`**: Inicializa pesos e *biases* aleatoriamente entre -0.5 e 0.5 para quebrar a simetria dos neurônios.
* **`predict`**: Realiza a passagem para frente (Forward).
* **`train_step`**: Implementa o coração do Backpropagation para uma única amostra.
* **`train`**: Gerencia o loop de épocas (10.000) e embaralha os dados (`shuffle`) para evitar enviesamento.
* **`test`**: Valida a rede e calcula a acurácia final.

---

## 4. FAQ - Possíveis Dúvidas da Banca

### Por que os pesos não podem começar com zero?

Se todos os pesos forem zero, todos os neurônios da camada oculta calcularão a mesma saída e receberão o mesmo gradiente. Isso impediria que a rede aprendesse diferentes características do dado (problema da simetria).

### Para que serve o Bias?

O *bias* permite que a função de ativação seja deslocada no eixo, permitindo que a rede aprenda limiares de decisão que não passam necessariamente pela origem (0,0).

### Por que utilizar 4 neurônios na camada oculta?

Este número foi escolhido via experimentação. Neurônios a menos poderiam não convergir (subajuste), e neurônios a mais poderiam causar sobreajuste (*overfitting*) em um conjunto pequeno de 20 amostras.

### O que é o Erro Médio Quadrático (MSE)?

É a métrica de custo utilizada:


$$MSE = \frac{1}{n} \sum (y_{alvo} - y_{calc})^2$$


Ele penaliza erros maiores de forma mais severa, auxiliando na estabilidade do gradiente.

### Por que usar 0.5 como limiar no teste?

Como a Sigmoide retorna valores entre 0 e 1, o valor 0.5 representa o ponto central de incerteza. Valores acima indicam maior probabilidade de pertencer à classe 1.0 (ex: frutas "boas").

---

## 5. Como Executar

Certifique-se de ter o Python 3 instalado. No terminal, execute:

```bash
python mlp.py

```

O programa exibirá a evolução do erro durante o treino e o relatório de acurácia final nos dados fornecidos pela disciplina.


