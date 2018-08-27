"""
		Nome: Igor Martinelli
		NUSP: 9006336
		SCC0275 - Introdução à Redes Neurais
		2018/2
		Trabalho 1: Implementação Adaline
"""
import numpy as np
import random
import glob
#Função responsável por realizar o treinamento da rede.
def adaline(datasets, weights, values, eta):
    #erro inicializado em 1.
    error = 1
    #enquanto o erro for diferente de 0, o laço roda até encontrar a configuração ideal de pesos.
    while(error != 0):
        error = 0
        for (dataset, value) in zip(datasets, values):
            #calcula a f_net, representada pela função hard limiter para o exemplo atual.
            f_net = forward(dataset, weights)
            #verifica se os pesos precisam ser atualizados, se sim, atualiza-os e incrementa a variável error.
            weights, error_aux = backwards(dataset, value, f_net, weights, eta)
            error += error_aux
    #retorna os pesos calculados.
    return weights

#Função responsável por aplicar a função hard limiter na variável predict.
def forward(x, weights):
    #calcula o valor previsto para o conjunto de dados.
    predict = np.dot(x, weights)
    #caso o valor seja >= 0, então retorna classe 1.
    if(predict >= 0):
        return 1
    #caso contrário, retorna classe -1.
    else:
        return -1

#Função responsável por calcular o erro e, se necessário, atualizar os pesos.
def backwards(x, y, y_hat, weights, eta):
    #cálculo do erro.
    E = y - y_hat
    #se o erro é diferente de 0, atualiza os pesos.
    if(E != 0):
        weights = weights + eta*x*E
    #retorna os pesos e o erro.
    return (weights, E)

#Função responsável por inicializar os pesos e direcionar os exemplos para a etapa de treinamento pela função adaline.
def adaline_train(train):
    weights = []
    for i in range(26):
        weights.append(random.uniform(-0.3, 0.3))

    value = []
    dataset = []
    for example in train:
        #lendo os conjuntos de treinamento e adicionando o 1 ao final para ser o fator multiplicador de theta.
        data = np.loadtxt(example)
        value.append(data[0, -1])
        dataset.append(np.append(data[:, 0:5].reshape(-1), 1))
        
    return adaline(dataset, weights, value, 0.1)

#Função responsável por realizar a etapa de teste a partir da rede previamente treinada.
def adaline_test(test, weights):
    counter = 0
    for example in test:
        #lendo o conjunto de treinamento e adicionando o 1 ao final para ser o fator multiplicador de theta.
        data = np.loadtxt(example)
        value = data[0, -1]
        dataset = np.append(data[:, 0:5].reshape(-1), 1)
        #predição dos valores.
        predict = forward(dataset, weights)
        #incrementando a variável counter responsável por retornar a acurácia.

        #print("Valor esperado: " + str(value))
        #print("Valor predito: " + str(predict))
        if(predict == value):
            counter += 1
    #retorno da acurácia do algoritmo.
    return (counter/len(test))

#Para um total de mil execuções, mede a acurácia média do algoritmo.
accuracys = []
for i in range(1):
    #adquirindo arquivos de treino e teste.
    examples = glob.glob("dataset/*.txt")
    #separando 2/3 para treino e 1/3 para teste.
    train = examples[0:int((2/3)*len(examples))]
    test = examples[int((2/3)*len(examples)):]
    #treinando a rede para encontrar os pesos ideais.
    weights = adaline_train(train)
    #validação do conjunto de treino.
    accuracys.append(adaline_test(test, weights))

print("Acurácia do algoritmo foi de: " + str(np.mean(accuracys)*100) + "%.")