# coding: utf-8
#Energy efficiency

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lerDados():
    #Abrindo o arquivo e lendo os dados
    dados = pd.read_csv('data.csv', sep = ',')

    #Alocando os dados, x1,x2,x3,x4,x5,x6
    x = dados.iloc[:, 0:8].values

    print ("--------------------------Lendo os dados de X----------------------")
    print (x)
    print ("--------------------------Fim Da Leitura de Dados de X-------------")

    #Dados do meu Y ideal
    y = dados['Y1'].values
    y = y.reshape((y.shape[0],1))

    print ("----------------------- ---Lendo os dados de Y----------------------")
    print (y)
    print ("--------------------------Fim Da Leitura de Dados de Y--------------")

    #Plot Gráfico Inicial
    plt.plot(x, y, '.')
    plt.plot(x, y)
    plt.show()

    return x, y

def calcularDados(x, y):
    #Iniciando o Teta com um vetor coluna
    teta = np.zeros((x.shape[1], 1))

    #Numero de Iterações
    maxIterate = 10000

    #precisao
    eta = 0.0000000000009

    #vetor de perda
    loss = []

    for i in range(maxIterate):
        #Calculo do y chapeu
        y_hat = x.dot(teta)

        #adicionando no vetor de perda
        loss.append(((y - y_hat) ** 2).sum())

        #calculando o gradiente
        gradiente = -x.T.dot(y-x.dot(teta))

        #Calculando e atualizando o valor do teta
        teta = teta - eta * gradiente

    print ("---------------Valores de Teta------------------")
    print (teta)
    print ("---------------Fim------------------------------")

    #Plot gráfico para saber se divergiu
    plt.plot(range(maxIterate), loss)
    plt.show()

    return teta

def main():
    x, y = lerDados()

    teta = calcularDados(x, y)

    print ("-------------------Taxa de Acurracia----------------")
    print (((x.dot(teta)).sum()/y.sum())*100)

    arq = open('y_hat.txt', 'w')
    arq.write(str(x.dot(teta)))
    arq.close()

    arq1 = open('erro.txt', 'w')
    arq1.write(str(y-(x.dot(teta))))
    arq.close()


main()