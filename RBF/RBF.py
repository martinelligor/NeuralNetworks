"""
		Nome: Igor Martinelli      Zoltán Hirata Jetsmen
		NUSP: 9006336              9293272
		SCC0275 - Introdução à Redes Neurais
		2018/2
		Exercício 3: RBF
"""
import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
"""
    readDataset: Function that reads and randomize the dataset, returning a namedtuple -> dataset.X and dataset.Y
"""
def readDataset(filename, y_columns):
    #reading the dataset.
    data = pd.read_csv(filename, index_col=False, header=None, sep='\t')
    #acquiring dataset data and class data.
    y = data.iloc[:,-1]
    y = np.array(y)
    X = data.iloc[:,:-1]
    X = np.array(X)
    #randomizing dataset.
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]
    #creating an alias to dataset -> dataset.X and dataset.Y
    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)
    return d

"""
    processing: Function that transform divides the dataset in train and test and transform the Y values in binary.
"""
def processing(dataset, percentage):
    #normalizing process.
    scaler = StandardScaler()
    scaler.fit(dataset.X)
    x = scaler.transform(dataset.X)
    #labelizing process.
    onehot_encoder = OneHotEncoder(sparse=False)
    y = dataset.Y.reshape(len(dataset.Y), 1)
    y = onehot_encoder.fit_transform(y)

    #computing the lenght of dataset.
    lenght = dataset.X.shape[0]

    #split dataset into train and test.
    x_train = x[0:int(percentage*lenght), :]
    y_train = y[0:int(percentage*lenght), :]

    x_test = x[int(percentage*lenght):, :]
    y_test = y[int(percentage*lenght):, :]
        
    #creating an alias to train and test set.
    dataset = namedtuple('datset', 'X Y')
    train = dataset(X=x_train, Y=y_train)
    test = dataset(X=x_test, Y=y_test)

    return train, test

"""
	sigmoid: função respons�vel pela representação da função sigmoide utilizada na rede MLP.
"""
def sigmoid(x):

    return 1/(1+np.exp(-x))

"""
    mlp_forward: Function that is responsible for the forward step, that applies the actual weight values on the net.
"""
def mlp_forward(x, hidden_weights, output_weights):

    f_net_h = []
    #apllying the weights on the hidden units.
    for i in range(len(hidden_weights)):
        #if is the first hidden unit.
        if i == 0:
            net = np.matmul(x,hidden_weights[i][:,0:len(x)].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        #if is the second or more hidden unit
        else:
            net = np.matmul(f_net_h[i-1],hidden_weights[i][:,0:len(f_net_h[i-1])].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        
        #store f_net of hidden layers.
        f_net_h.append(f_net) 

    #computing the net function to the output layer.
    net = np.matmul(f_net_h[len(f_net_h)-1],output_weights[:,0:len(f_net_h[len(f_net_h)-1])].transpose()) + output_weights[:,-1]
        
    f_net_o = sigmoid(net)
    
    return f_net_o, f_net_h

"""
    mlp_backward: Function that is responsible for the backpropagation step, which corresponds to the updating of weights.
"""
def mlp_backward(dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, eta, hidden_units, n_classes):

    x = dataset.X[j,:]
    y = dataset.Y[j,:]
    #measuring the error.
    error = y - f_net_o

    delta_o = error*f_net_o*(1-f_net_o)
    
    #computing the delta for the hidden units.
    delta_h = []
    for i in range(len(hidden_units)-1, -1, -1):

        if(i == len(hidden_units)-1):
            w_o = output_weights[: ,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta_o, w_o))
        else:
            w_o = hidden_weights[i+1][:,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta, w_o))

        delta_h.insert(0,delta)
    
    #computing the delta and updating weights for the output layer.
    delta_o = delta_o[:, np.newaxis]
    f_net_aux = np.concatenate((f_net_h[len(hidden_units)-1],np.ones(1)))[np.newaxis, :]
    output_weights = output_weights - -2*eta*np.matmul(delta_o, f_net_aux)
    
    #updating the weights for the hidden layers.
    for i in range(len(hidden_units)-1, -1, -1):
        delta = delta_h[i][:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]    

        if i == 0:
            x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, x_aux)
        else:
            f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, f_net_aux)

    #measuring the error.
    error = sum(error*error)

    #return the updated weights and the new error.
    return hidden_weights, output_weights, error
"""
    testing: Function that is responsible to realize the tests.
"""
def mlp_testing(train, test, hidden_weights, output_weights):
    counter = 0
    for i in range(test.X.shape[0]):
        y_hat, q = mlp_forward(test.X[i,:], hidden_weights, output_weights)
        y_hat = np.argmax(y_hat)
        y = np.argmax(test.Y[i,:])
        if y == y_hat:
            counter += 1

    return counter/test.X.shape[0]

"""
    MLP: function that is responsible to initialize weights, check conditions to construct net.
"""
def MLP(dataset, hidden_layers ,hidden_units, n_classes, eta, data_ratio, threshold):
    #checking conditions to construct net.
    if(len(hidden_units) != hidden_layers):
        print("The parameter hidden_units must have its length the same value that hidden_layers.")
        return

    #acquiring the train and test set.
    train, test = processing(dataset, data_ratio)

    #initializing the weights of the hidden layers.
    hidden_weights = []
    for i in range(hidden_layers):
        if(i == 0):
            aux = np.zeros((hidden_units[i], dataset.X.shape[1] + 1))
        else:
            aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))
    
        hidden_weights.append(aux)
    
    #filling the hidden layers weight values with a normal distribution between -1 and 1.
    for i in range(hidden_layers):
        for j in range(hidden_units[i]):
            if(i == 0):
                for k in range(dataset.X.shape[1] + 1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)
            else:
                for k in range(hidden_units[i-1]+1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)

    #initializing and filling the weights values of output layer.
    output_weights = np.zeros((n_classes, hidden_units[len(hidden_units)-1]+1))

    for i in range(n_classes):
        for j in range(hidden_units[hidden_layers-1]+1):
            output_weights[i][j] = random.uniform(-1, 1)

    sum_errors = 2* threshold
    epoch = 0
    while(sum_errors > threshold):
        sum_errors = 0
        for i in range(train.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(train.X[i, :], hidden_weights, output_weights)
            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error= mlp_backward(train, i, hidden_weights, output_weights, f_net_o,
                                                                f_net_h, eta, hidden_units, n_classes)
            sum_errors += error
        epoch += 1

        if(epoch % 100 == 0):
            print (sum_errors)
        
    return mlp_testing(train, test, hidden_weights, output_weights)

"""
    Função hidden_train: função responsável por calcular a distância para cada cluster e aplicar a função gaussiana de base radial.
"""
def hidden_train(x, n_clusters, clusters, sigma):
    distances = np.zeros(n_clusters)

    for i in range(n_clusters):
        #calculando a distância euclidiana para cada cluster i.
        distances[i] = np.sqrt(np.sum(((x - clusters[i])**2)))
    #applying radial basis function
    distances = np.exp(-(((distances**2))/(2*(sigma**2))))
    #envia para a camada escondida o cálculo da distância do exemplo x para cada cluster.
    return distances
"""
    Função forward: função responsável por aplicar os pesos da camada de saida juntamente com as distancias obtidas para cada cluster.
"""
def forward(x, output_weights):
    #função de ativação = função identidade. Concatena com 1 p/ adição do theta.
    x_aux = np.concatenate((x, np.ones(1)))
    return np.matmul(x_aux, output_weights.T) 
"""
    Função backward: função responsável pela atualização dos pesos da camada de saída.
"""
def backward(dataset, eta, n_classes, output_weights, entry, f_net_o, distance):
    y = dataset.Y[entry]
    #calcula o erro.
    error = y - f_net_o
    #atualiza os pesos da camada de saída.
    output_weights += eta*np.matmul(error.reshape(n_classes,1),np.append(distance,1).reshape(1,n_classes+1))

    return output_weights, error
"""
    Função testing: função responsável pela etapa de teste da rede RBF.
"""
def testing(dataset, output_weights, n_clusters, clusters, sigma):
    counter = 0
    #para cada entrada de teste..
    for entry in range(dataset.X.shape[0]):
        #calcula a distancia para cada cluster
        distances = hidden_train(dataset.X[entry], n_clusters, clusters, sigma)
        #aplica as distancias juntamente com os pesos obtidos
        y_hat = forward(distances, output_weights)
        #computa a saida esperada e a obtida
        y_hat = np.argmax(y_hat)
        y = np.argmax(dataset.Y[entry])
        #compara as saidas, se igual -> soma 1 acerto.
        if (y == y_hat):
            counter += 1

    return (counter/dataset.X.shape[0])
"""        
    Função RBF: função responsável por todas as etapas de execução da rede RBF.
"""
def RBF(dataset, n_classes, eta, data_ratio, epochs, sigma):
    train, test = processing(dataset, data_ratio)
    n_clusters = n_classes
    clusters = np.zeros((n_clusters, dataset.X.shape[1]))

    # calculating the clusters.
    for i in range(1, n_clusters+1):
        clusters[i-1] = train.X[np.argmax(train.Y,axis=1)+1 == i].mean(axis=0)
    
    hidden_units = n_clusters
    #initializing and filling the weights values of output layer.
    output_weights = np.zeros((n_clusters,n_classes+1))

    #inicialização dos pesos da camada de saída com distribuição uniforme de -1 a 1.
    for i in range(n_clusters):
        for j in range(n_classes+1):
            output_weights[i][j] = random.uniform(-1, 1)

    #treina a rede de acordo com o número de épocas.
    for i in range(epochs):
        error = 0
        #para cada entrada de treino...
        for entry in range(train.X.shape[0]):
            #calcula a distancia do exemplo para cada cluster.
            distances = hidden_train(train.X[entry], n_clusters, clusters, sigma)
            #calcula a saida esperada com base nos pesos atuais.
            f_net_o = forward(distances, output_weights)
            #atualiza os pesos e computa o erro
            output_weights, erro = backward(train, eta, n_classes, output_weights, entry, f_net_o, distances)
            error += sum(erro*erro)
    #etapa de teste.
    return testing(test, output_weights, n_clusters, clusters, sigma)

#Teste das acurácias da rede RBF.
rbf_acc = 0
for i in range(10):
    rbf_acc += RBF(readDataset("seeds_dataset.txt", 1), 3, 0.01, 0.7, 500, 1.9)

print("Acurácia média RBF: " + str(rbf_acc/10))

#Teste de acurácia da rede MLP (O teste com a média foi calculado com a execução de 10 vezes do programa, pois as vezes o mesmo cai num mínimo local e demora muito para sair.)
mlp_acc = MLP(readDataset("seeds_dataset.txt", 1), 1, [3], 3, 0.05, 0.7, 0.1)
print("Acurácia MLP: " + str(mlp_acc))
