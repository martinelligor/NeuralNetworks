"""
		Nome: Igor Martinelli      Zoltán Hirata Jetsmen
		NUSP: 9006336              9293272
		SCC0275 - Introdução à Redes Neurais
		2018/2
		Exercício 5: PCA
"""
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import pandas as pd
import numpy as np

def readDataset():
    iris = datasets.load_iris()
    iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])

    dataset = namedtuple('datset', 'X Y')
    d = dataset(X=iris.iloc[:, :-1], Y=iris.iloc[:, -1])

    return d

# PCA(Principal Component Analysis) -> Considerado como uma feature selection, onde reduz a dimensionalide dos dados de entrada
def PCA():
    dataset = readDataset()

    #Passo 1: Centralizar os dados em torno do ponto 0. Caso as features possuem unidades de medidas diferentes, devemos dividir o resultado pela standard deviation.
    scaled = StandardScaler().fit_transform(dataset.X.astype(float))
    
    #Passo 2: Calcular a covariancia da matrix de dados, onde a covariância indica o grau de interdependência númerica entre duas variáveis
    covMatrix = (np.corrcoef(scaled.astype(float).T))

    #Passo 3: Calcular os autovalores e autovetores da matrix de covariancia
    w, v = np.linalg.eig(covMatrix)

    #Verificar o quanto de informação pode ser atribuido para cada componente
    percentage = (w/sum(w))*100
    print('Informação atribuida para cada componente: ', percentage)

    eig_pairs = [(np.abs(w[i]), v[:,i]) for i in range(len(w))]

    # Concatena horizontalmente as features.
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                          eig_pairs[1][1].reshape(4,1),
                          eig_pairs[2][1].reshape(4,1),
                          eig_pairs[3][1].reshape(4,1)))

    X = scaled.dot(matrix_w)

    df = pd.DataFrame(data=X, columns=['Principal component 1', 'Principal component 2',
                                       'Principal component 3', 'Principal component 4'])
    df['target'] = dataset.Y
    sns.pairplot(data=df, hue='target')
    plt.show()

PCA()