
"""
Este script trata el punto 1 del trabajo.

Usar algún algoritmo de clustering sobre todos los datos sin emplear las etiquetas para obtener un primer análisis de los datos. 
El objetivo consiste en descubrir posibles patrones en los datos y hasta qué punto estos son compatibles con las etiquetas.

"""


# importamos las librerías y funciones necesarias

import numpy as np
import sklearn
import librosa
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, adjusted_rand_score,  mutual_info_score, normalized_mutual_info_score,  adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score,  fowlkes_mallows_score,silhouette_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

np.random.seed(123)
seed = np.random.seed(123)

# función importar y transformar los datos etiquetados

def importar_y_transformar_datos(n_mfcc,n_mels=128,n_components=0.99,verbose=0):
    """
    Argumentos:
    * n_mfcc: número de elementos MFCC que queremos seleccionar
    * n_mels: número de bandas de Mel a generar
    * n_components: porcentaje de la varianza explicada que queremos obtener con la transformación PCA

    Salidas:
    * data: matriz de datos original
    * X: matriz de datos transformada
    * labs: etiquetas de los datos
    """
    data = np.load('./datos/adata5.npy')
    nrow,ncol = data.shape
    names = ['Door knock', 'Mouse click', 'Keyboard typing', 'Door, wood creaks', 'Can opening','Washing machine']
    c = len(names) # número de clases
    nc = nrow//c # muestras por clase. // aproxima al menor entero
    labs = np.int16(np.kron(np.arange(c), np.ones(nc))) # generamos las etiquetas

    # convertimos al espacio de Mel-frequecny cepstral coefficients (MFCC)
    srate = 22050 # sampling rate
    l = librosa.feature.mfcc(y=data[0,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten().size
    data_mfcc = np.zeros([nrow,l])
    for i in range(nrow):
        data_mfcc[i,:] = librosa.feature.mfcc(y=data[i,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten()

    # análisis de componentes principales
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=seed)    
    pca.fit(data_mfcc)
    X = pca.transform(data_mfcc)
    if verbose==1:
        print("\n")
        print("Dimensiones originales %d, dimensiones tras transformación %d" % (data.shape[1],X.shape[1]))

    return [data,X,labs]

# función importar y transformar los datos NO etiquetados

def importar_y_transformar_datos_no_etiquetados(n_mfcc,n_mels=128,n_components=0.99,verbose=0):
    """
    Argumentos:
    * n_mfcc: número de elementos MFCC que queremos seleccionar
    * n_mels: número de bandas de Mel a generar
    * n_components: porcentaje de la varianza explicada que queremos obtener con la transformación PCA

    Salidas:
    * data: matriz de datos original
    * X: matriz de datos transformada
    """
    data = np.load('./datos/udata.npy')
    nrow,ncol = data.shape

    # convertimos al espacio de Mel-frequecny cepstral coefficients (MFCC)
    srate = 22050 # sampling rate
    l = librosa.feature.mfcc(y=data[0,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten().size
    data_mfcc = np.zeros([nrow,l])
    for i in range(nrow):
        data_mfcc[i,:] = librosa.feature.mfcc(y=data[i,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten()

    # análisis de componentes principales
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=seed)    
    pca.fit(data_mfcc)
    X = pca.transform(data_mfcc)
    if verbose==1:
        print("Dimensiones originales %d, dimensiones tras transformación %d" % (data.shape[1],X.shape[1]))

    return [data,X]


# función importar y transformar TODOS LOS DATOS

def importar_y_transformar_TODOS_datos(n_mfcc,n_mels=128,n_components=0.99,verbose=0):
    """
    Argumentos:
    * n_mfcc: número de elementos MFCC que queremos seleccionar
    * n_mels: número de bandas de Mel a generar
    * n_components: porcentaje de la varianza explicada que queremos obtener con la transformación PCA

    Salidas:
    * data: matriz de datos original
    * X: matriz de datos transformada
    """
    data_labeled = np.load('./datos/adata5.npy')
    data_no_labeled = np.load('./datos/udata.npy')
    data = np.concatenate((data_labeled,data_no_labeled),axis=0)
    nrow = data.shape[0]

    # convertimos al espacio de Mel-frequecny cepstral coefficients (MFCC)
    srate = 22050 # sampling rate
    l = librosa.feature.mfcc(y=data[0,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten().size
    data_mfcc = np.zeros([nrow,l])
    for i in range(nrow):
        data_mfcc[i,:] = librosa.feature.mfcc(y=data[i,:],sr=srate,n_mfcc=n_mfcc,n_mels=n_mels).flatten()

    # análisis de componentes principales
    pca = sklearn.decomposition.PCA(n_components=n_components, random_state=seed)    
    pca.fit(data_mfcc)
    X = pca.transform(data_mfcc)
    if verbose==1:
        print("\n")
        print("Dimensiones originales %d, dimensiones tras transformación %d" % (data.shape[1],X.shape[1]))

    return [data,X]



# Función para calcular los índices de evaluación del clustering

def siluetas_test(X,y,y_pred,range_n_clusters,comparation_true_labels=1):

    """
    Argumentos:
    * X:  matriz de datos 
    * y: etiquetas verdaderas 
    * y_pred: lista con etiquetas predichas por el algoritmo de cada iteración
    * range_n_clusters: número de clusters de cada iteración

    Salida:
    * l: lista con todos los índices calculados
    """

    # listas vacías para almacenar los valores de los índices
    silouettes = []
    calinskis  = []
    aris = []
    mis = []
    # nmis = []
    # amis = []
    hs = []
    cs = []
    vs = []
    fmis = []
    cms = []
    
    for n_clusters in range_n_clusters:
        # predicción de etiquetas
        cluster_labels = y_pred[n_clusters - range_n_clusters[0]]

        # índices que no usan las etiquetas verdaderas        
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_h = calinski_harabasz_score(X,cluster_labels)
        silouettes.append(silhouette_avg)
        calinskis.append(calinski_h)

        # índices que usan las etiquetas verdaderas
        if comparation_true_labels==1: 
            
            aris.append(adjusted_rand_score(y, cluster_labels))
            mis.append(mutual_info_score(y, cluster_labels))
            # nmis.append(normalized_mutual_info_score(y, cluster_labels))
            # amis.append(adjusted_mutual_info_score(y, cluster_labels))
            hs.append(homogeneity_score(y, cluster_labels))
            cs.append(completeness_score(y, cluster_labels))
            vs.append(v_measure_score(y, cluster_labels))
            fmis.append(fowlkes_mallows_score(y,cluster_labels))
            cms.append(contingency_matrix(y,cluster_labels))

            # retorno de la función
            l = [silouettes, calinskis, aris, mis, hs, cs, vs, fmis, cms, range_n_clusters]
            # silouettes, calinskis, aris, mis, nmis, amis, hs, cs, vs, fmis, cms, range_n_clusters
        else:
            # retorno de la función
            l = [silouettes, calinskis, range_n_clusters]

    return l






# función para dibujar curvas clustering

def curvas_diagnostico(l,comparation_true_labels=1, legend=1, title="",ax1 = 0):
    """
    Los argumentos son la salida de la función anterior
    """
    if comparation_true_labels==1:
        silouettes, calinskis, aris, mis, hs, cs, vs, fmis, cms, range_n_clusters = l
        #  silouettes, calinskis, aris, mis, nmis, amis, hs, cs, vs, fmis, cms, range_n_clusters
    else:
        silouettes, calinskis, range_n_clusters = l

    if ax1 == 0 :
        fig, ax1 = plt.subplots(figsize=(6,6))
    else:
        ax1 = ax1

    ax1.plot(range_n_clusters,silouettes,'b-',label='silhouette')
    ax1.set_ylim(min(0,ax1.get_ylim()[0]) , 1)
    ax1.grid()
    ax1.set_xticks(range_n_clusters)
    ax1.set_xlabel("Número de clusters")

    if comparation_true_labels==1:
        ax1.set_ylabel("Otras medidas",color='b')
        ax1.plot(range_n_clusters,aris,label='ARI',lw=4,alpha=.5)
        ax1.plot(range_n_clusters,mis,label='MI')
        # ax1.plot(range_n_clusters,nmis,'o-',label='NMI')
        # ax1.plot(range_n_clusters,amis,'o-',label='AMI')
        ax1.plot(range_n_clusters,hs,'--',label='Homogeneity')
        ax1.plot(range_n_clusters,cs,'--',label='Completeness')
        ax1.plot(range_n_clusters,vs,'--',label='V-measure',lw=4)
        ax1.plot(range_n_clusters,fmis,'k:',label='FMI',lw=3)

    ax2=ax1.twinx()
    ax2.plot(range_n_clusters,calinskis,'r--',linewidth=4)    
    ax2.set_xlabel("Número de clusters")
    ax2.set_ylabel("Calinski-Harabazt",color='r')

    if legend==1: ax1.legend()

    ax1.set_title(title) 

    ax2.set_ylim( min(0,plt.ylim()[0]) ,  max(1,plt.ylim()[1])    )
    ax2.grid()
    ax2.set_xticks(range_n_clusters)
    # ax2.set_title('Comparación con etiquetas reales') # Comparing to ground-truth

    # plt.show()

    return ax1



# plotear índices del clustering. 
# LA SILUETA MEDIA DE CADA GRUPO. ESTA NO LA USAMOS EN EL TRABAJO
#  calculado con la función anterior. 
# Solo válida si  comparation_true_labels=1 en la función anterior

def dibujar_siluetas_clusters(X,y_pred, l, title=""):
    """
    Argumentos:
    * X: matriz de datos transformados
    * y_pred: predicciones a evaluar
    * l: indices calculados con la función siluetas_test y la opción comparation_true_labels=1.

    Retorno. 
    * plots
    """

    # desempaquetamos argumentos
    silhouettes, calinskis, aris, mis, hs, cs, vs, fmis, cms, range_n_clusters = l

    # ajustar el número de columnas del plot
    m = len(range_n_clusters) // 2
    if len(range_n_clusters) % 2 == 0:
        m = m
    else:
        m = m + 1

    fig,axs = plt.subplots(ncols=2,nrows=m,figsize=(15,m*3))
    o = 0 # índice de cada iteración. Para seleccionar elementos de silhouettes
    for n_clusters,ax in zip(range_n_clusters,axs.ravel()): # esto puede ser útil

        # predicción de etiquetas
        cluster_labels = y_pred[n_clusters - range_n_clusters[0]]
        
        # El coeficiente de silueta varía en el intervalo [-1, 1]
        ax.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.Spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title(title)
        ax.set_subtitle("Silhouette para clustering "+str(n_clusters))
        ax.set_xlabel("Coeficiente Silhouette")
        ax.set_ylabel("Etiqueta del cluster")

        silhouette_avg = silhouettes[o]
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])        

        o += 1


    plt.tight_layout() # para que los ejes se lean bien

