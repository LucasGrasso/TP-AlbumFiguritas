import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats


def generar_paquete(_figus_total, _figus_paquete, _weights):
    return (np.random.choice(list(np.arange(0,_figus_total)), _figus_paquete,replace=False)).tolist() #Generar randoms ponerados con np' replace=False para evitar repes y p=weights para distintas probabilidades
     #Le saco el param p=_weights porque al ser las weights random complica mucho el llenado y alarga el runtime

def esta_lleno(_album):
    albumShape = _album.shape
    return np.sum(_album)//(albumShape[0]*albumShape[1])

def cantorFunction(x1,x2): #La encontre en google y es para relacionar dos ints, mas rapido que iterar por todos los tuples(Creo?)
    return (((x1+x2)*(x1+x2+1))/2)+x2

def cuantas_figus(_figus_total,_figusxPaquete,_cantColeccionistas,_figusWeights):
    figusAdquiridas = []
    album = np.full(
                    shape=(_figus_total,_cantColeccionistas),
                    fill_value=0,
                    dtype=int
                    )
    repes={}
    for coleccionista in range(0,_cantColeccionistas): #Creamos para guardar coleccionistas
        repes[coleccionista] = []   
    contPaquetes = 0

    faltantes = [set(np.arange(0,_figus_total).flatten()) for _ in range(_cantColeccionistas)]
    
    while not esta_lleno(album):
        yaProcesados=set() #Lo utilizamos para evitar operar sobre los mismos dos coleccionistas mas de una vez
        for coleccionista in range (0,_cantColeccionistas): #Iteramos sobre todos los coleccionistas para chequear todos con todos
            coleccionistas = np.arange(0,_cantColeccionistas)
            coleccionistas = np.delete(coleccionistas, np.where(coleccionistas==coleccionista))
            while(len(coleccionistas) != 0):#Logica de intercambiado de figus
                coleccionistaSeleccionado = coleccionistas[random.randint(0,len(coleccionistas)-1)] #Seleccionamos un coleccionista random y lo borramos de la lista
                coleccionistasPaired = cantorFunction(coleccionista,coleccionistaSeleccionado)
                if not coleccionistasPaired in yaProcesados:
                    yaProcesados.add(coleccionistasPaired)
                    coleccionistas = np.delete(coleccionistas, np.where(coleccionistas==coleccionistaSeleccionado))
                    if len(repes[coleccionistaSeleccionado]) > 0 and len(repes[coleccionista]) > 0:
                        for figuRepeSeleccionado in repes[coleccionistaSeleccionado]: #iteramos sobre todas las repes del seleccionado
                            if figuRepeSeleccionado in faltantes[coleccionista]: 
                                for figuRepeColeccionista in repes[coleccionista]: #Nos fijamos si alguna repe del coleccionista le sirve al seleccionado
                                    if figuRepeColeccionista in faltantes[coleccionistaSeleccionado] and figuRepeColeccionista != figuRepeSeleccionado and figuRepeSeleccionado in faltantes[coleccionista]:
                                        album[figuRepeSeleccionado][coleccionista] = 1; album[figuRepeColeccionista][coleccionistaSeleccionado] = 1
                                        repes[coleccionistaSeleccionado].remove(figuRepeSeleccionado)
                                        repes[coleccionista].remove(figuRepeColeccionista)
                                        faltantes[coleccionistaSeleccionado].remove(figuRepeColeccionista) #Producir el intercambio, eliminar de repes, faltantes y hacer 1 el valor en album
                                        faltantes[coleccionista].remove(figuRepeSeleccionado)
                                
        for coleccionista in range (0,_cantColeccionistas):
            contPaquetes += 1
            for figu in generar_paquete(_figus_total,_figusxPaquete,_figusWeights):   
                if album[figu][coleccionista] == 0: 
                    album[figu][coleccionista] = 1
                    faltantes[coleccionista].remove(figu)
                    figusAdquiridas.append(figu)
                else:
                    repes[coleccionista].append(figu)
    return (contPaquetes,figusAdquiridas)


def generateRandomWeights(_tamaño):
    weights = np.random.random(_tamaño)
    weights /= weights.sum()
    return weights

def insertionSort(arr):
    for i in range(1, len(arr)):
        p = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > p:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = p
    return arr            #Extra5

def mediana(arr):
    i = len(arr)//2
    if len(arr) %2 != 1:
        return arr[i]
    return (arr[i- 1] + arr[i]) / 2
    
def promedio(tamaño,repeticiones,figusPaquete,cantColeccionistas):
    res = []
    figuWeights = generateRandomWeights(tamaño)
    for _ in range(0,repeticiones):
        res.append(cuantas_figus(tamaño,figusPaquete,cantColeccionistas,figuWeights))
    prom = np.mean(res) #Obtener Promedio
    #print("Mediana de respuesta es: " + str(mediana(insertionSort(res))))
    print(f"El promedio es: {prom}")
    return prom

def chequearProb(tamaño,repeticiones,figusPaquete,cantColeccionistas):
    numAChequar = int(input("Que numero desea chequear? "))
    contMenorANum= 0;
    weights = [(1/tamaño)]*tamaño
    for _ in range(0,repeticiones):
        if cuantas_figus(tamaño,figusPaquete,cantColeccionistas,weights)[0] <= numAChequar:
            contMenorANum+=1
    prob = (contMenorANum/repeticiones)
    print(f"Prob de completar con menos de {numAChequar} paquetes: {prob}")
    return prob

def LinearRegPred(): #Extra1
    x = np.array([850, 1150, 900, 950, 1000, 1100]).reshape((-1, 1))
    y = np.array([0.36, 0.85, 0.45, 0.56, 0.66, 0.86])
    model = LinearRegression().fit(x, y)
    XTotal = (np.arange(0,2000)).reshape((-1, 1))
    Pred_Y = model.predict(XTotal)
    print('prediction:', Pred_Y, sep='\n')
    #Scatter Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('% De completar el album con menos de x figus', fontsize=15)   
    ax.set_xlabel('Paquetes Comprados', fontsize=12)
    ax.set_ylabel('% de Completar', fontsize=12)
    ax.axis([0, 2000, 0, 1])
    #Plot Model
    XTotal = (np.arange(0,2000)); XTotal.reshape((-1, 1))
    ax.plot(XTotal,Pred_Y,color='r', label='LinearRegModel')
    ax.plot(x, y,'o', color="b", label='True Values')

    plt.legend()
    plt.show()
    return

def getDistrib(_figusTotales,_figusList):
    figusTotales = np.array(_figusList)
    figusSeries = pd.Series(figusTotales)
    pd.DataFrame(figusTotales).plot(kind="density",
                                      figsize=(10,10),
                                      xlim=(-1,_figusTotales));
    
    mayorDensidad = [0,0]
    for x in range(-1, _figusTotales,1):
        densityX = stats.uniform.pdf(x, loc=0, scale=_figusTotales)
        if densityX >= mayorDensidad[0]:
            mayorDensidad[0] = densityX
            mayorDensidad[1] = x
    print(f"La mayor Densidad fue: {mayorDensidad[0]} y fue en x={mayorDensidad[1]}")
        
    return
     
tamAlbum = int(input("Ingrese el tamaño del album: "))
rep = int(input("Ingrese el numero de repeticiones a promediar: "))
figXPaq = int(input("Ingrese el numero de figuritas por paquete: ")) #Inputs
cantCol = int(input("Ingrese la cantidad de collecionistas en el problema: "))
    
promedio(tamAlbum, rep, figXPaq, cantCol)