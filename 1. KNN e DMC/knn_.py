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

def train_method(train):
    return train

def evaluate_algorithm(dataset, algorithm, n_folds, k):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = [f for f in folds if f is not fold]
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        test_set = np.array(test_set)
        predicted = algorithm(train_set, test_set, k)
        actual = [row[-1] for row in fold]
        classes = set(list(dataset[:,-1].astype('int')))
        n_classes = len(classes)
        cm, accuracy = accuracy_metric(actual, predicted, n_classes)
        #print(accuracy)
        scores.append(accuracy)
    
    return scores

def predict(x_test, train, k):
    distances = []
    ordered = []
    positions = []
    classes = []

    train = np.array(train)
    for row in train:
        distance = math.sqrt(sum(pow(row[:-1] - x_test, 2)))
        distances.append(distance)
        
    ordered = distances[:]
    ordered.sort()

    i = 0
    for order in ordered:
        if(i == k):
            break
        
        positions.append(distances.index(order))
        i = i + 1

    for pos in positions:
        classes.append(train[pos][-1])
    
    a = np.array(classes)
    a = a.astype('int')
    counts = np.bincount(a)
    max_value = np.argmax(counts)
    return max_value

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split

def train_test_split(dataset, test_size):
    import random
    dataset_split = list()
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    qtd_total_data = len(dataset_copy)
    print qtd_total_data
    qtd_test = int(test_size*qtd_total_data)
    print qtd_test
    qtd_train = qtd_total_data - qtd_test

    train, test = list(), list()
    while len(test) < qtd_test:
        index = random.randrange(len(dataset_copy))
        test.append(np.array(dataset_copy.pop(index)))

    train = dataset_copy[:] #Training set

    train = np.array(train)
    return train, np.array(test)

def knn(train, test, k):
    predictions = []
    for row in test:
        prediction = predict(row[:-1], train, k)
        predictions.append(prediction)

    return(predictions)

def accuracy_metric(actual, predicted, n_classes):
    correct = 0
    classes = len(set(actual))
    #print actual
    #print classes
    confusion_matrix = np.zeros((n_classes,n_classes), int)    
    for i in range(len(actual)):
        actual[i] = int(actual[i])
        confusion_matrix[actual[i]][predicted[i]] = confusion_matrix[actual[i]][predicted[i]] + 1
        if actual[i] == predicted[i]:
            correct += 1
    
    acc = correct / float(len(actual)) * 100.0
    return confusion_matrix, acc

def graph(n_row, n_column, pts, gr, gr_train, colors_pts, colors_gr, markers_pts):
    if n_row == 1 and n_column == 1:
        plt.title('Superficie de Decisao.')
        for m in range(len(colors_pts)):
            plt.scatter(gr[0][m][:,0], gr[0][m][:,1], color = colors_gr[m])
            plt.scatter(gr_train[0][m][:,0], gr_train[0][m][:,1], marker = '.', color = colors_pts[m])
            plt.scatter(pts[0][m][:,0], pts[0][m][:,1], marker = markers_pts[m], color = colors_pts[m])

    else:
        f, t = plt.subplots(n_row, n_column, sharex='col', sharey='row')
        f.suptitle('Superficies de Decisao com atributos combinados par a par.')
        i = 0
        for l in t:
            for g in l:
                for m in range(len(colors_pts)):
                    print m
                    print gr_train[i][m][:,0], gr_train[i][m][:,1], colors_pts[m]
                    g.scatter(gr[i][m][:,0], gr[i][m][:,1], color = colors_gr[m])
                    g.scatter(gr_train[i][m][:,0], gr_train[i][m][:,1], marker = '.', color = colors_pts[m])
                    g.scatter(pts[i][m][:,0], pts[i][m][:,1], marker = markers_pts[m], color = colors_pts[m])
                i = i + 1    
    plt.show()


def graphic(conj, conj1, k):
    
    new_train_datasets = []

    for i in range(len(conj[0])-1):
        for j in range(i+1, len(conj[0])-1):
            new_dataset = []
            X = conj[:,i]
            Y = conj[:,j]
          
            for c in range(len(conj)):
                new_dataset.append([X[c],Y[c],conj[c][-1]])
            
            new_dataset = np.array(new_dataset)
            new_train_datasets.append(new_dataset)
            
    new_train_datasets = np.array(new_train_datasets)

    new_tests = [] #6 conj de testes

    for i in range(len(conj1[0])-1):
        for j in range(i+1, len(conj1[0])-1):
            new_test = []
            X = conj1[:,i]
            Y = conj1[:,j]
            
            for c in range(len(conj1)):
                new_test.append(np.array([X[c],Y[c],None]))
            
            new_tests.append(new_test)

    new_tests = np.array(new_tests)

    pts = []
    colors_pts = ['darkred','darkblue','darkgreen']
    colors_pts_ = []
    markers_pts = ['*','+','x']
    markers_pts_ = []
    for i in range(len(new_tests)):
        pred = knn(new_train_datasets[i], new_tests[i], k)
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        new_tests[i] = np.concatenate((new_tests[i][:,:-1],pred), axis = 1)
        pts_pontos = []
        for c in classes:
            points = new_tests[i][np.where(new_tests[i][:,-1] == c)]
            pts_pontos.append(points)
        pts.append(pts_pontos)

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
    gr_train = []
    colors_gr = ['salmon', 'skyblue', 'lightgreen']
    colors_gr_ = []
    tests_all = []
    for train in new_train_datasets:
        pred = knn(train, data_test_all, k)
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        dataset = np.concatenate((data_test_all[:,:-1],pred), axis = 1)
        gr_points =  []
        gr_points_train =  []
        train = np.array(train)
        for c in classes:
            points = dataset[np.where(dataset[:,-1] == c)]
            points_train = train[np.where(train[:,-1] == c)]
            gr_points.append(points)
            gr_points_train.append(points_train)

        
    
        gr.append(gr_points)
        gr_train.append(gr_points_train)
        tests_all.append(dataset)
    
    for c in classes:
        colors_gr_.append(colors_gr[c])

    n_row, n_column = 1, 1
    if len(gr) == 6:
        n_row, n_column = 2, 3
    elif len(gr) == 15:
        n_row, n_column = 3, 5

    #graph(n_row, n_column, pts, gr, colors_pts_, colors_gr_, markers_pts_)
    graph(n_row, n_column, pts, gr, gr_train, colors_pts_, colors_gr_, markers_pts_)


if __name__ == "__main__":

    entered = input('Choose the base if you want run the KNN:\n(a)-Iris;\n(b)-Vertebral Column;\n(c)-Base Artificial.\n')

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
  
    models = []
    k_vectors = []
    accs = []
    accs_t = []
    times_train = []
    times_test = []
    for realizacoes in range(20):
        train, test = train_test_split(dataset, 0.2)

        time_train_start = time.time()
        train = train_method(train)
        time_train_end = time.time()
        time_train = time_train_end - time_train_start
        times_train.append(time_train)
        
        accuracies_k = []
        for k in range(1,25):
            scores = evaluate_algorithm(train, knn, 5, k)
            #print('Scores: %s' % scores)
            mean = sum(scores)/float(len(scores))
            std = np.std(scores)
            #print("Media das taxas de acerto:", mean)
            #print("Desvio padrao:",std)
            #print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
            accuracies_k.append(mean)
            
        print accuracies_k
        accs.append(max(accuracies_k))
        print max(accuracies_k)
        k = accuracies_k.index(max(accuracies_k)) + 1
        print "Value of k:", k
        k_vectors.append(k)
        time_test_start = time.time()
        predicted = knn(train, test, k)
        time_test_end = time.time()
        time_test = time_test_end - time_test_start
        times_test.append(time_test)
        actual = [row[-1] for row in test]
        n_classes = len(set(list(dataset[:,-1])))
        mc, accuracy = accuracy_metric(actual, predicted, n_classes)
        print mc
        print "Acuracy of test:"
        print accuracy
        accs_t.append(accuracy)
        models.append((train, test, k))
        #scores.append(accuracy)
    
    print("Valores de k: ", k_vectors)
    print("Acuracias de treinamento: ", accs)
    print("Acuracias de teste: ", accs_t)
    mean_realizations = sum(accs)/float(len(accs))
    print("Media das acuracias de treinamento:", mean_realizations)
    print("Media das acuracias de test:", mean_realizations)
    print("Desvio padrao - treinamento:", np.std(accs))
    print("Desvio padrao - teste:", np.std(accs_t))
    print("Tempo medio de treinamento: ", sum(times_train)/float(len(times_train)))
    print("Tempo medio de teste: ", sum(times_test)/float(len(times_test)))
    dist = np.array(accs) - mean_realizations
    dist = np.abs(dist)
    dist = list(dist)
    min_dist = min(dist)
    index = dist.index(min_dist)
    graphic(models[index][0], models[index][1], models[index][2])

    plt.scatter(k_vectors, accs)
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy Train')
    plt.title('Accuracy x Value of k - KNN')
    plt.show()

    plt.scatter(k_vectors, accs_t)
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy Test')
    plt.title('Accuracy x Value of k - KNN')
    plt.show()