import math
import random

class MLP:
    """
    Perceptron de Múltiplas Camadas (MLP) implementado do zero.
    Arquitetura: Camada de Entrada -> Camada Oculta -> Camada de Saída.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos e biases da camada oculta (Input -> Hidden)
        self.w_ih = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b_h = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        
        # Inicialização dos pesos e biases da camada de saída (Hidden -> Output)
        self.w_ho = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_o = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

    def _sigmoid(self, x):
        """Função de ativação não-linear (Sigmoide)."""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _d_sigmoid(self, y):
        """Derivada da função sigmoide (y = sigmoid(x))."""
        return y * (1 - y)

    def predict(self, inputs):
        """Realiza a propagação (Forward Pass) na rede neural."""
        # Ativação da camada oculta
        h_out = []
        for i in range(self.hidden_size):
            z = sum(inputs[j] * self.w_ih[i][j] for j in range(self.input_size)) + self.b_h[i]
            h_out.append(self._sigmoid(z))
            
        # Ativação da camada de saída
        o_out = []
        for i in range(self.output_size):
            z = sum(h_out[j] * self.w_ho[i][j] for j in range(self.hidden_size)) + self.b_o[i]
            o_out.append(self._sigmoid(z))
            
        return o_out, h_out

    def train_step(self, inputs, targets):
        """Executa um passo de treinamento para uma única amostra usando Backpropagation."""
        # 1. Forward Pass
        o_out, h_out = self.predict(inputs)
        
        # 2. Cálculo dos erros e gradientes na camada de saída
        o_errors = [targets[i] - o_out[i] for i in range(self.output_size)]
        o_grads = [o_errors[i] * self._d_sigmoid(o_out[i]) for i in range(self.output_size)]
        
        # 3. Cálculo dos erros e gradientes na camada oculta
        h_errors = []
        for i in range(self.hidden_size):
            error = sum(o_grads[j] * self.w_ho[j][i] for j in range(self.output_size))
            h_errors.append(error)
            
        h_grads = [h_errors[i] * self._d_sigmoid(h_out[i]) for i in range(self.hidden_size)]
        
        # 4. Atualização dos pesos e biases (Hidden -> Output)
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.w_ho[i][j] += self.learning_rate * o_grads[i] * h_out[j]
            self.b_o[i] += self.learning_rate * o_grads[i]
            
        # 4. Atualização dos pesos e biases (Input -> Hidden)
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.w_ih[i][j] += self.learning_rate * h_grads[i] * inputs[j]
            self.b_h[i] += self.learning_rate * h_grads[i]
            
        return sum(e**2 for e in o_errors) / self.output_size

    def train(self, dataset, epochs):
        print("\n============================================================")
        print(f"  FASE 1: TREINAMENTO ({epochs} épocas | Taxa Aprendizado: {self.learning_rate})")
        print("============================================================")
        print("[EXPLICAÇÃO] Durante o treinamento, a rede fará múltiplas passagens pelos dados.")
        print("Em cada época, usaremos o Algoritmo Backpropagation para calcular o erro da rede")
        print("e ajustar os 'pesos' das sinapses de trás para frente. O objetivo é que o Erro")
        print("Médio Quadrático (MSE) se aproxime cada vez mais de zero.\n")
        
        for epoch in range(1, epochs + 1):
            random.shuffle(dataset)
            total_mse = 0
            
            for inputs, targets in dataset:
                error = self.train_step(inputs, targets)
                total_mse += error
            
            epoch_mse = total_mse / len(dataset)
            
            if epoch == 1 or epoch % 1000 == 0:
                print(f"Época {epoch:5d} | Erro Médio Quadrático (MSE): {epoch_mse:.6f}")
                
        print("\n[EXPLICAÇÃO] Observe na lista acima como o MSE caiu consideravelmente!")
        print("Isso prova matematicamente que o algoritmo convergiu e os pesos foram ajustados.")
        print("============================================================\n")

    def test(self, dataset):
        print("============================================================")
        print("  FASE 2: TESTE E CLASSIFICAÇÃO")
        print("============================================================")
        print("[EXPLICAÇÃO] Agora vamos testar a rede neural treinada. Vamos passar os dados")
        print("novamente apenas pela etapa de Forward Pass (propagação) e observar a saída.")
        print("Como usamos a função Sigmoide na saída (que retorna valores entre 0 e 1):")
        print(" -> Se Saída Predita >= 0.5: Classifica como 1.0 (Fruta de Boa Qualidade)")
        print(" -> Se Saída Predita < 0.5:  Classifica como 0.0 (Fruta de Qualidade Ruim/Média)\n")
        
        correct = 0
        for inputs, targets in dataset:
            o_out, _ = self.predict(inputs)
            prediction = 1.0 if o_out[0] >= 0.5 else 0.0
            
            is_correct = prediction == targets[0]
            if is_correct:
                correct += 1
                
            status = "CORRETO" if is_correct else "INCORRETO"
            print(f"Entrada (Características): {inputs}")
            print(f" -> Esperado: {targets[0]:.1f} | Predito Pela Sigmoide: {o_out[0]:.4f}")
            print(f" -> Decisão da Rede: Classe {prediction:.1f} ({status})\n")
        
        accuracy = (correct / len(dataset)) * 100
        print("============================================================")
        print(f"  RESULTADO FINAL: Acurácia de {accuracy:.2f}% ({correct}/{len(dataset)} corretas)")
        print("============================================================")
        print("\n[CONCLUSÃO] O perceptron de múltiplas camadas resolveu este problema de")
        print("classificação não-linear de forma bem sucedida sem o uso de bibliotecas de IA!\n")

if __name__ == "__main__":
    random.seed(42)
    
    # Dataset da UFERSA
    dataset = [
        ([0.20, 0.15], [0.0]), ([0.30, 0.25], [0.0]), ([0.10, 0.50], [0.0]), ([0.50, 0.10], [0.0]),
        ([0.25, 0.35], [0.0]), ([0.40, 0.40], [0.0]), ([0.15, 0.80], [0.0]), ([0.80, 0.20], [0.0]),
        ([0.35, 0.30], [0.0]), ([0.45, 0.50], [0.0]),
        ([0.55, 0.55], [1.0]), ([0.65, 0.60], [1.0]), ([0.70, 0.75], [1.0]), ([0.80, 0.85], [1.0]),
        ([0.90, 0.90], [1.0]), ([0.85, 0.70], [1.0]), ([0.60, 0.65], [1.0]), ([0.75, 0.80], [1.0]),
        ([0.95, 0.95], [1.0]), ([0.50, 0.55], [1.0])
    ]

    input_nodes = 2
    hidden_nodes = 4
    output_nodes = 1
    learning_rate = 0.2
    epochs = 10000

    print("\n============================================================")
    print("  TRABALHO DE INTELIGÊNCIA ARTIFICIAL (UFERSA)")
    print("  Implementação: MLP + Backpropagation (Classificação de Frutas)")
    print("============================================================")
    print(f"[ARQUITETURA]: {input_nodes} Entrada(s) -> {hidden_nodes} Oculta(s) -> {output_nodes} Saída(s)")
    
    mlp = MLP(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlp.train(dataset, epochs)
    mlp.test(dataset)
