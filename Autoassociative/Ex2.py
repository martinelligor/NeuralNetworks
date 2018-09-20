import pandas as pd
import numpy as np
import random
from collections import namedtuple

"""
	readDataset: função responsável por ler o dataset e randomizar as linhas do mesmo a fim de sortear os exemplos.
"""
def readDataset(filename):

    data = pd.read_csv(filename, index_col=False, header=None)
    
    y = data.iloc[:,len(data.columns)-10: len(data.columns)]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-10]
    X = np.array(X)

    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)

    return d
"""
	sigmoid: função responsável pela representação da função sigmoide utilizada na rede MLP.
"""
def sigmoid(x):

    return 1/(1+np.exp(-x))

"""
	mlp_forward: função responsável por realizar a etapa de forward da rede MLP. Tal função aplica os pesos atuais nos neurônios da rede a fim de atualizá-los.
""" 
def mlp_forward(x, hidden_weights, output_weights):

    f_net_h = []

    for i in range(len(hidden_weights)):
        if i == 0:
            net = np.matmul(x,hidden_weights[i][:,0:len(x)].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        else:
            net = np.matmul(f_net_h[i-1],hidden_weights[i][:,0:len(f_net_h[i-1])].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        
        f_net_h.append(f_net) 

    net = np.matmul(f_net_h[len(f_net_h)-1],output_weights[:,0:len(f_net_h[len(f_net_h)-1])].transpose()) + output_weights[:,-1]
        
    f_net_o = sigmoid(net)
 
    return f_net_o, f_net_h

"""
	mlp_backward: função responsável por realizar a etapa de backpropagation da rede MLP. Tal etapa é responsável pela atualização dos pesos da camada de saída e escondida.
"""
def mlp_backward(dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes):

    x = dataset.X[j,:]
    y = dataset.Y[j,:]

    error = y - f_net_o

    delta_o = error*f_net_o*(1-f_net_o)
    
    delta_h = []

    for i in range(len(hidden_units)-1, -1, -1):

        if(i == len(hidden_units)-1):
            w_o = output_weights[: ,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta_o, w_o))
        else:
            w_o = hidden_weights[i+1][:,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta, w_o))

        delta_h.insert(0,delta)
        
    delta_o = delta_o[:, np.newaxis]
    f_net_aux = np.concatenate((f_net_h[len(hidden_units)-1],np.ones(1)))[np.newaxis, :]
    output_weights = output_weights - -2*alpha*np.matmul(delta_o, f_net_aux)
    
    for i in range(len(hidden_units)-1, -1, -1):
        delta = delta_h[i][:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]    

        if i == 0:
            x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*alpha*np.matmul(delta, x_aux)
        else:
            f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*alpha*np.matmul(delta, f_net_aux)

    error = sum(error*error)

    return hidden_weights, output_weights, error 

"""
	MLP: função responsável pela inicialização dos pesos e treinamento da rede.
"""
def MLP(dataset, hidden_layers ,hidden_units, n_classes, threshold, alpha):

    if(len(hidden_units) != hidden_layers):
        print("The parameter hidden_units must have its length the same value that hidden_layers.")
        return

    hidden_weights = []

    for i in range(hidden_layers):
        if(i == 0):
            aux = np.zeros((hidden_units[i], dataset.X.shape[1] + 1))
        else:
            aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))

        hidden_weights.append(aux)
    
    for i in range(hidden_layers):
        for j in range(hidden_units[i]):
            if(i == 0):
                for k in range(dataset.X.shape[1] + 1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)
            else:
                for k in range(hidden_units[i-1]+1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)

    output_weights = np.zeros((n_classes, hidden_units[len(hidden_units)-1]+1))

    for i in range(n_classes):
        for j in range(hidden_units[hidden_layers-1]+1):
            output_weights[i][j] = random.uniform(-1, 1)

    sum_errors = 2* threshold
    epoch = 0
    while(sum_errors > threshold):
        sum_errors = 0
        for i in range(dataset.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(dataset.X[i, :], hidden_weights, output_weights)
            
            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error = mlp_backward(dataset, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes)

            sum_errors += error
    
        epoch += 1
        
        if(epoch % 500 == 0):
            print(sum_errors)

    for i in range(10):
        print(dataset.X[i,:])
        y_hat, f_net_h = mlp_forward(dataset.X[i, :], hidden_weights, output_weights)
        y_hat = np.round(y_hat, 2)
        print(y_hat)
        print()

MLP(readDataset("dataset.txt"), 1, [4], 10, 0.001, 0.1)
