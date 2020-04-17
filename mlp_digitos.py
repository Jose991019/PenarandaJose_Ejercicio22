import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.preprocessing import StandardScaler
import sklearn.neural_network
import sklearn.model_selection
from sklearn.model_selection import train_test_split

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

loss = []
F1_train = []
F1_test = []
for i in range(1,21):
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(i), 
                                               max_iter=1200)
    mlp.fit(X_train, y_train)
    loss.append(mlp.loss_)
    F1_train.append(sklearn.metrics.f1_score(y_train, mlp.predict(X_train), average='macro'))
    F1_test.append(sklearn.metrics.f1_score(y_test, mlp.predict(X_test), average='macro'))
    
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.plot(range(1,21),loss)
plt.xlabel('Número de neuronas')
plt.ylabel('Loss')
plt.title('Núm. óptimo: 5 neuronas')
plt.subplot(1,2,2)
plt.plot(range(1,21),F1_train,label = 'Train')
plt.plot(range(1,21),F1_test,label = 'Test')
plt.legend()
plt.xlabel('Número de neuronas')
plt.ylabel('F1 score')
plt.title('Núm. óptimo: 5 neuronas')
plt.savefig('loss_f1.png')

mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(5), 
                                               max_iter=1000)
mlp.fit(X_train, y_train)
plt.figure(figsize = (25,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    scale = np.max(mlp.coefs_[0])
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, 
                       vmin=-scale, vmax=scale)
    plt.title('Neurona {}'.format(i+1))
plt.savefig('neuronas.png')