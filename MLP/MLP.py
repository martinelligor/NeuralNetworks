"""
		Nome: Igor Martinelli      Zoltán Hirata Jetsmen
		NUSP: 9006336              9293272
		SCC0275 - Introdução à Redes Neurais
		2018/2
		Projeto 1: MLP
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
def readDataset(filename, y_collumns):
    #reading the dataset.
    data = pd.read_csv(filename, index_col=False, header=None)
    #acquiring dataset data and class data.
    y = data.iloc[:,len(data.columns)-y_collumns: len(data.columns)]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-y_collumns]
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
    processing: Function that transform divides the dataset in train and test and transform the Y values in binary, OneHotEnconding
"""
def processing(dataset, percentage, method):
    #if method is classification, normalize and labelize data.
    if(method == "C"):
        #normalizing process.
        scaler = StandardScaler()
        scaler.fit(dataset.X)
        x = scaler.transform(dataset.X)
        #labelizing process.
        onehot_encoder = OneHotEncoder(sparse=False)
        y = dataset.Y.reshape(len(dataset.Y), 1)
        y = onehot_encoder.fit_transform(y)
    #if method is regression.
    else:
        x = dataset.X
        #normalizing output to improve regression method.
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(dataset.Y)
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
    sigmoid: Function that applies the sigmoid function, used in the backpropagation step.
""" 
def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
    mlp_forward: Function that is responsible for the forward step, that applies the actual weight values on the net.
"""
def mlp_forward(x, hidden_weights, output_weights,method):

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
def mlp_backward(dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, eta, hidden_units, alpha, momentum_h, momentum_o, n_classes, method):

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
    output_weights = output_weights - -2*eta*np.matmul(delta_o, f_net_aux) + momentum_o
    momentum_o = - -2*eta*np.matmul(delta_o, f_net_aux)
    
    #updating the weights for the hidden layers.
    for i in range(len(hidden_units)-1, -1, -1):
        delta = delta_h[i][:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]    

        if i == 0:
            x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, x_aux) + momentum_h[i]
            momentum_h[i] = - -2*eta*np.matmul(delta, x_aux)
        else:
            f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*eta*np.matmul(delta, f_net_aux) + momentum_h[i]
            momentum_h[i] = - -2*eta*np.matmul(delta, f_net_aux)

    #measuring the error.
    error = sum(error*error)

    #return the updated weights, the new error and the momentum parameters.
    return hidden_weights, output_weights, error, momentum_h, momentum_o

"""
    testing: Function that is responsible to realize the tests for the classification and regression methods
             for different datasets.
"""
def testing(train, test, hidden_weights, output_weights, method):
    #if method is classification.
    if(method == "C"):
        counter = 0

        for i in range(test.X.shape[0]):
            y_hat, q = mlp_forward(test.X[i,:], hidden_weights, output_weights, method)
            y_hat = np.argmax(y_hat)
            y = np.argmax(test.Y[i,:])
            if y == y_hat:
                counter += 1

        return counter/test.X.shape[0]
    #if method is regression.
    else:
        sum_errors = 0

        for i in range(test.X.shape[0]):
            y_hat, q = mlp_forward(test.X[i,:], hidden_weights, output_weights, method)
            error = test.Y[i,:] - y_hat
            error = error*error
            sum_errors += sum(error)
        
        return sum_errors/test.X.shape[0]

"""
    MLP: function that is responsible to initialize weights, check conditions to construct net.
"""
def MLP(dataset, hidden_layers ,hidden_units, n_classes, epochs, eta, alpha, data_ratio, method):
    #checking conditions to construct net.
    if(len(hidden_units) != hidden_layers):
        print("The parameter hidden_units must have its length the same value that hidden_layers.")
        return

    if(method != "R" and method != "C"):
        print("The parameter method must be R (Regression) or C (Classification).")
        return
    
    #acquiring the train and test set.
    train, test = processing(dataset, data_ratio, method)

    #initializing the weights of the hidden layers.
    momentum_o = 0
    momentum_h = []
    hidden_weights = []
    for i in range(hidden_layers):
        if(i == 0):
            aux = np.zeros((hidden_units[i], dataset.X.shape[1] + 1))
        else:
            aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))
    
        hidden_weights.append(aux)
        momentum_h.append(aux)

    
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

    epoch = 0
    for epoch in range(epochs):
        sum_errors = 0
        for i in range(train.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(train.X[i, :], hidden_weights, output_weights, method)
            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error, momentum_h, momentum_o = mlp_backward(train, i, hidden_weights, output_weights, 
                                                                                        f_net_o, f_net_h, eta, hidden_units, alpha, 
                                                                                        momentum_h, momentum_o, n_classes, method)
            sum_errors += error
        epoch += 1

        if method == "R":
            sum_errors = sum_errors/train.X.shape[0]
        
    #computing the measure for the chosen method.
    return testing(train, test, hidden_weights, output_weights, method)

"""
    mlp_pipeline: function that is responsible for all the experiments.
"""
def mlp_pipeline():
    #lists of parameters variation.
    etas = [1e-1, 1e-2, 1e-3]
    data_ratios = [0.7, 0.8, 0.9]
    alphas = [1, 2, 5]
    epochs = [200, 500, 1000]
    datasets = ['C', 'R']
    layers = [1, 2]
    units_1_c = [[2], [3], [4]]
    units_2_c = [[2, 1], [2, 3], [3, 4]]
    units_1_r = [[20], [30], [40]]
    units_2_r = [[20, 10], [20, 30], [30, 40]]


    #reading the two datasets.
    reg_dt = readDataset("default_features_1059_tracks.txt", 2)
    class_dt = readDataset("wine.csv", 1)

    df = pd.DataFrame(columns=['Hidden Layers', 'HiddenL1 Units', 'HiddenL2 Units', 'Epochs', 'Alpha', 'Eta', 'Train ratio', 'Test ratio', 'Accuracy'], dtype=float)
    #loops to execute all the experiments.
    for dataset in datasets:
        for layer in layers:
            for epoch in epochs:
                for alpha in alphas:
                    for eta in etas:
                        for data_ratio in data_ratios:
                            if(layer == 1):
                                for units in range(len(units_1_c)):
                                    if(dataset == 'C'):
                                        measure = MLP(class_dt, layer, units_1_c[units], 3, epoch, eta, alpha, data_ratio, dataset)
                                        df.loc[len(df)+1] = [layer, units_1_c[units][0], None,  epoch, alpha, eta, data_ratio, 1-data_ratio, measure]
                                    else:
                                        measure = MLP(reg_dt, layer, units_1_r[units], 2, epoch, eta, alpha, data_ratio, dataset)
                                        df.loc[len(df)+1] = [layer, units_1_r[units][0], None,  epoch, alpha, eta, data_ratio, 1-data_ratio, measure]
                            else:
                                for units in range(len(units_2_c)):
                                    if(dataset == 'C'):
                                        measure = MLP(class_dt, layer, units_2_c[units], 3, epoch, eta, alpha, data_ratio, dataset)
                                        df.loc[len(df)+1] = [layer, units_2_c[units][0], units_2_c[units][1],  epoch, alpha, eta, data_ratio, 1-data_ratio, measure]
                                    else:
                                        measure = MLP(reg_dt, layer, units_2_r[units], 2, epoch, eta, alpha, data_ratio, dataset)
                                        df.loc[len(df)+1] = [layer, units_2_r[units][0], units_2_c[units][1],  epoch, alpha, eta, data_ratio, 1-data_ratio, measure]
        if(dataset == 'C'):
            df.to_csv('Classification.csv')
            df = None
        else:
            df.to_csv('Regression.csv')


mlp_pipeline()