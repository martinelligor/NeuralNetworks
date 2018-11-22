import random
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import RegularPolygon

# Função responsável por realizar o treino do modelo.
def build_architecture(dataset, layer_size, n_epochs, sigma_0):
    tal = n_epochs/np.log(sigma_0)
    # inicializa a camada de kohonen com valores numa distribuição uniforme entre -0.5 e 0.5.
    grid = np.random.uniform(-0.5, 0.5, size=(layer_size, layer_size, len(dataset.columns[:-1])))
    
    return grid, tal, 
            
# Função responsável por treinar o modelo.
def train_model(dataset, grid, sigma_0, alpha_0, n_epochs):   
    for epoch in range(n_epochs):
        for entry in dataset.iloc[:, :-1].values:
            # calculando a distância euclidiana para cada neurônio da rede
            distances = np.sqrt(np.sum(np.power((entry - grid[:, :]), 2), axis=2))
            row = int(distances.argmin()/layer_size)
            col = int(distances.argmin()%layer_size)
            # calcula as variáveis de atualização dos pesos
            e = np.exp(-epoch/tal)
            sigma = sigma_0*e
            alpha = alpha_0*e
            # distancia topológica
            for (i, j) in np.ndindex(layer_size, layer_size):
                # cálculo da radial
                radial = np.exp(-(np.sum((i-row)**2 + (j-col)**2))/(2*np.power(sigma,2)))
                grid[i, j] += alpha*radial*(entry - grid[i, j])
                
# Função responsável por plotar o resultado exibido por cor mostrando qual classe os neurônios da rede são mais propensos a classificar.
def plot_model(dataset, grid, layer_size):
    colors = {0:'red', 1:'blue', 2:'green'}
    color_map = np.zeros((layer_size, layer_size), dtype=str)

    for (i,j) in np.ndindex(layer_size, layer_size):
        dist = np.sqrt(np.sum(np.power(grid[i, j] - dataset.iloc[:,:-1].values, 2), axis=1))
        color_map[i, j] = (colors.get(int(dataset.iloc[:,-1][dist.argmin()])))

    # Mapeando as coordenadas dos hexágonos.
    coord = []
    for x in np.arange(10):
        for i in range(10):
            if(x%2 == 0):
                coord.append([x, i, -i])
            else:
                if(i!=0):
                    coord.append([x, i, -i+i/i])

    # Coordenadas horizontais.
    hcoord = [c[0] for c in coord]
    # Coordenadas verticais.
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    colors = color_map.reshape(-1)
    fig, ax = plt.subplots(1, figsize=(15,15))
    ax.set_aspect('equal')
    # Adicionando os hexágonos.
    for (i, j, color) in zip(hcoord, vcoord, colors):
        hex = RegularPolygon((i, j), numVertices=6, radius=2. / 3., 
                             orientation=np.radians(30),
                             facecolor = color,
                             alpha=0.5, edgecolor='k')
        ax.add_patch(hex)
        plt.rcParams.update({'font.size': 22})
        plt.axis('off')

    # legendas.
    r = mlines.Line2D([], [], color='red', marker='H',
                          markersize=20, label='Class 0')
    b = mlines.Line2D([], [], color='blue', marker='H',
                          markersize=20, label='Class 1')
    g = mlines.Line2D([], [], color='green', marker='H',
                          markersize=20, label='Class 2')

    plt.legend(handles=[r, b, g])
    ax.plot()
    plt.show()
    
# lendo o conjunto de dados.
wine = datasets.load_wine()
wine = pd.DataFrame(data= np.c_[wine['data'], wine['target']], columns= wine['feature_names'] + ['target'])

# escolhendo os parâmetros do modelo.
sigma_0 = 5
alpha_0 = 0.1
layer_size = 10
n_epochs = 1000

# construindo a arquitetura do modelo com base nos parâmetros.
grid, tal = build_architecture(wine, layer_size, n_epochs, sigma_0)
# treinando o modelo.
train_model(wine, grid, sigma_0, alpha_0, n_epochs)
# plotando a classe dos centróides.
plot_model(wine, grid, layer_size)