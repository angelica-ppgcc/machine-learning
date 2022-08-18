import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import time

def normalization(X):
    for j in range(len(X[0,:])):
        X[:,j] = (X[:,j] - min(X[:,j]))/(max(X[:,j]) - min(X[:,j]))
    
    return X
    

def load_data(dataset):

    array = dataset.values
    X = array[:,:-1]
    Y = array[:,-1]
    for i in range(len(Y)):
        if Y[i] == 'Iris-setosa' or Y[i] == 'DH':
            Y[i] = 0
        elif Y[i] == 'Iris-versicolor' or Y[i] == 'SL':
            Y[i] = 1
        else:
            Y[i] = 2

    return X, Y

def train_method(train_set):
    
    classes = set(train_set[:,-1])

    groups = []
    for c in classes:
        group = train_set[np.where(train_set[:,-1] == c)]
        group = group[:,:-1]
        groups.append(group)

    centroids = []
    for g in groups:
        mean = (np.sum(g, axis = 0))/len(g)
        mean = list(mean)
        centroids.append(mean)

    return np.array(centroids)

def predict(x_test, centroids):

    distances = []
    
    for centroid in centroids:
        distance = math.sqrt(sum(pow(centroid - x_test, 2)))
        distances.append(distance)
        
    distance_min = min(distances)
    class_predicted = distances.index(distance_min)

    return class_predicted

def train_test_split(dataset, test_size):
    import random
    dataset_split = list()
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    qtd_total_data = len(dataset_copy)
    qtd_test = int(test_size*qtd_total_data)
    qtd_train = qtd_total_data - qtd_test

    train, test = list(), list()
    while len(test) < qtd_test:
        index = random.randrange(len(dataset_copy))
        test.append(np.array(dataset_copy.pop(index)))

    train = dataset_copy[:] #Training set

    train = np.array(train)
    return train, np.array(test)

def dmc(test, centroids):

    predictions = []
    for row in test:
        prediction = predict(row[:-1], centroids)
        predictions.append(prediction)

    return(predictions)

def accuracy_metric(actual, predicted, classes):
    correct = 0
    confusion_matrix = np.zeros((classes,classes), int)    
    for i in range(len(actual)):
        actual[i] = int(actual[i])
        confusion_matrix[actual[i]][predicted[i]] = confusion_matrix[actual[i]][predicted[i]] + 1
        if actual[i] == predicted[i]:
            correct += 1
    
    acc = correct / float(len(actual)) * 100.0
    return confusion_matrix, acc


def graph(n_row, n_column, pts, trs, gr, colors_pts, colors_gr, markers_pts, new_centroids):
    if n_row == 1 and n_column == 1:
        plt.title('Superficie de Decisao.')
        for m in range(len(colors_pts)):
            plt.scatter(gr[0][m][:,0], gr[0][m][:,1], color = colors_gr[m])
            plt.scatter(trs[0][m][:,0], trs[0][m][:,1], marker = '.', color = colors_pts[m])
            plt.scatter(pts[0][m][:,0], pts[0][m][:,1], marker = markers_pts[m], color = colors_pts[m])

        plt.scatter(new_centroids[0][:,0], new_centroids[0][:,1], marker = "o", color = 'black')

    else:
        f, t = plt.subplots(n_row, n_column, sharex='col', sharey='row')
        f.suptitle('Superficies de Decisao com atributos combinados par a par.')
        i = 0
        for l in t:
            for g in l:
                for m in range(len(colors_pts)):
                    g.scatter(gr[i][m][:,0], gr[i][m][:,1], color = colors_gr[m])
                    g.scatter(trs[i][m][:,0], trs[i][m][:,1], marker = '.', color = colors_pts[m])
                    g.scatter(pts[i][m][:,0], pts[i][m][:,1], marker = markers_pts[m], color = colors_pts[m])

                g.scatter(new_centroids[i][:,0], new_centroids[i][:,1], marker = "o", color = 'black')
                i = i + 1    
    plt.show()
    
def graphic(conj0, conj, centroids):
    
    new_centroids = [] #6 conj de centroids

    for i in range(len(centroids[0])):
        for j in range(i+1, len(centroids[0])):
            c1 = centroids[:,i]
            
            c2 = centroids[:,j]
            new_centroid = []

            for c in range(len(centroids)):
                new_centroid.append([c1[c],c2[c]])
            
            new_centroids.append(new_centroid)

    new_centroids = np.array(new_centroids)

    new_tests = [] #6 conj de testes
    new_trains = []

    for i in range(len(conj[0])-1):
        for j in range(i+1, len(conj[0])-1):
            new_test = []
            X = conj[:,i]
            Y = conj[:,j]
            
            for c in range(len(conj)):
                new_test.append(np.array([X[c],Y[c],None]))
            
            new_tests.append(new_test)

    new_tests = np.array(new_tests)


    for i in range(len(conj0[0])-1):
        for j in range(i+1, len(conj0[0])-1):
            new_train = []
            X_ = conj0[:,i]
            Y_ = conj0[:,j]
            
            for c in range(len(conj0)):
                new_train.append(np.array([X_[c],Y_[c],conj0[c][-1]]))
            
            new_trains.append(new_train)

    new_trains = np.array(new_trains)

    pts = []
    trs = []
    colors_pts = ['darkred','darkblue','darkgreen']
    colors_pts_ = []
    markers_pts = ['*','+','x']
    markers_pts_ = []

    for i in range(len(new_tests)):
        pred = dmc(new_tests[i], new_centroids[i])
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        new_tests[i] = np.concatenate((new_tests[i][:,:-1],pred), axis = 1)
  
        pts_pontos = []
        pts_trains = []
        for c in classes:
            points = new_tests[i][np.where(new_tests[i][:,-1] == c)]
            trains = new_trains[i][np.where(new_trains[i][:,-1] == c)]
            pts_pontos.append(points)
            pts_trains.append(trains)

        pts.append(pts_pontos)
        trs.append(pts_trains)

    for c in classes:
        colors_pts_.append(colors_pts[c])
        markers_pts_.append(markers_pts[c])
        

#Conjunto de Dados de todos os pontos
    X1_test = []
    X2_test = []
    data_test_all = []
    for x1 in range(101):
        for x2 in range(101):
            _x1 = float(x1)/100
            _x2 = float(x2)/100
            data_test_all.append(np.array([_x1,_x2, None]))
        
    data_test_all = np.array(data_test_all)
    gr = []
    colors_gr = ['salmon', 'skyblue', 'lightgreen']
    colors_gr_ = []

    for centroids in new_centroids:
        pred = dmc(data_test_all, centroids)
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        dataset = np.concatenate((data_test_all[:,:-1],pred), axis = 1) 
        gr_points =  []
        for c in classes:    
            points = dataset[np.where(dataset[:,-1] == c)]
            gr_points.append(points)
        
        gr.append(gr_points)
    
    for c in classes:
        colors_gr_.append(colors_gr[c])
    
    n_row, n_column = 1, 1
    if len(gr) == 6:
        n_row, n_column = 2, 3
    elif len(gr) == 15:
        n_row, n_column = 3, 5

    graph(n_row, n_column, pts, trs, gr, colors_pts_, colors_gr_, markers_pts_, new_centroids)

  

if __name__ == "__main__":

    entered = input('Choose the base if you want run the DMC:\n(a)-Iris;\n(b)-Vertebral Column;\n(c)-Base Artificial.\n')

    if(entered == 'a'):    
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pd.read_csv('iris.csv', names = names)
        X, Y = load_data(dataset)
    elif(entered == 'b'):
        dataset = pd.read_table('column_3C.dat', sep = " ")
        X, Y = load_data(dataset)
    else:
        dataset = pd.read_csv('base_artificial2.csv', sep = " ")
        array = dataset.values
        X = array[:,:-1]
        Y = array[:,-1]

    X = normalization(X)

    Y = np.reshape(Y, (len(Y),1))
    dataset = np.concatenate((X,Y), axis = 1)

    k_vectors = []
    scores = []
    models = []
    times_train = []
    times_test = []
    for realizacoes in range(20):
        
        train, test = train_test_split(dataset, 0.2)

        time_train_start = time.time()
        centroids = train_method(train)
        time_train_end = time.time()

        time_train = time_train_end - time_train_start
        times_train.append(time_train)

        time_test_start = time.time()
        predicted = dmc(test, centroids)
        time_test_end = time.time()

        time_test = time_test_end - time_test_start
        times_test.append(time_test)

        actual = [row[-1] for row in test]
        n_classes = len(set(list(dataset[:,-1])))
        mc, accuracy = accuracy_metric(actual, predicted, n_classes)
        #print('Scores: %s' % scores)
        #print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
        print("Taxa of acerts - test:")
        print(accuracy)
        print("Confusion Matrix:")
        print(mc)
        scores.append(accuracy)
        models.append((train, test, centroids))
        #scores.append(accuracy)
    
    std = np.std(scores)
    print("Desvio padrao:",std)
    mean = sum(scores)/float(len(scores))
    print("Accuracy-test:", mean)
    dist = np.array(scores) - mean
    dist = np.abs(dist)
    dist = list(dist)
    min_dist = min(dist)
    index = dist.index(min_dist)

    print("Tempo medio de treinamento:", sum(times_train)/float(len(times_train)))
    print("Tempo medio de teste:",  sum(times_test)/float(len(times_test)))
    graphic(models[index][0], models[index][1], models[index][2])