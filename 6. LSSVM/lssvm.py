import pandas as pd
import numpy as np
from numpy.linalg import inv, det, solve
import math
import matplotlib.pyplot as plt
import time

class BaysianClassifierWithRejection:

    def __init__(self, kernel = 'linear'):
        self.kernel = kernel
    
    def fit(self, train, gamma):
        classes = train[:,-1]
        self.train = train
        self.classes = classes
        self.attributs = train[:,:-1]
        #print(self.classes * self.classes.T) 
        
        classes_transpose = classes.reshape((len(classes), 1))
        y = (classes_transpose * self.classes)

        attributs_transpose = self.attributs.transpose()

        if self.kernel == 'linear':
            w = np.dot(self.attributs, attributs_transpose)
        
        elif self.kernel == 'quadratico':
            w = pow(np.dot(self.attributs, attributs_transpose) + 1, 2)
        
        elif self.kernel == 'rbf':
            self.sigma = np.var(train)
            n = len(self.attributs)
        
            w = np.zeros((n,n), dtype = np.float64)
            for i in range(n):
                for j in range(n):
                    w[i][j] = np.exp(-np.dot(self.attributs[i] - self.attributs[j].T, (self.attributs[i] - self.attributs[j].T).T)/pow(self.sigma,2))

        kernel = (classes_transpose * self.classes) * w + (1/float(gamma)) * np.eye(len(train[:,0]))

        line = np.concatenate(([0], self.classes), axis=0)
        line = line.reshape((1, len(line)))
        rest = np.concatenate((classes_transpose, kernel), axis = 1)
        total = np.concatenate((line, rest), axis = 0)
        A = total
        b = np.concatenate(([0], np.ones(len(classes))), axis = 0)

        A = np.array(A, dtype='float')
        b = np.array(b, dtype='float')
        
        
        x = solve(A,b.T)
        
        self.bias = x[0]
        self.alphas = x[1:]
        
        return total, x



    def prediction(self, test):
        attributs_train = self.train[:,:-1]
        attributs_test = test[:,:-1]
        classes_train = self.train[:,-1]

        attributs_train_transpose = attributs_train.transpose()
        
        if self.kernel == 'linear':
            K = np.dot(attributs_train_transpose.T, attributs_test.T)
        elif self.kernel == 'quadratico':
            K = pow(np.dot(attributs_train_transpose.T, attributs_test.T)+1, 2)
        
        elif self.kernel == 'rbf':
            n = len(attributs_train)
            m = len(attributs_test)

            K = np.zeros((n,m), dtype = np.float64)
            for i in range(n):
                for j in range(m):
                    K[i][j] = np.exp(-np.dot(attributs_train[i] - attributs_test[j].T, (attributs_train[i] - attributs_test[j].T).T)/pow(self.sigma,2))
        #c = 2
        #K = pow(1+np.dot(attributs_train_transpose.T, attributs_test.T)/c, len(attributs_test[0]))
    
        aux = (self.alphas * classes_train)
        aux = np.reshape(aux, (1,len(aux)))
        
        f = np.sum(K * np.tile(aux, (len(attributs_test), 1)).T, axis = 0) + self.bias
    
        y = np.sign(f.tolist())
        return y

    def accuracy(self, actual, predicted, n_classes):
        
        correct = 0
        confusion_matrix = np.zeros((n_classes,n_classes), int) 
        #print(confusion_matrix)   
        for i in range(len(actual)):
            actual[i] = int(actual[i])
            predicted[i] = int(predicted[i])
            if actual[i] == -1:
                actual[i] = 0
            if predicted[i] == -1:
                predicted[i] = 0
           
            confusion_matrix[actual[i]][int(predicted[i])] = confusion_matrix[actual[i]][int(predicted[i])] + 1
            if actual[i] == predicted[i]:
                correct += 1
        
        acc = correct / float(len(actual)) * 100.0
        return confusion_matrix, acc


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

def normalization(X):
    for j in range(len(X[0,:])):
        X[:,j] = (X[:,j] - min(X[:,j]))/(max(X[:,j]) - min(X[:,j]))
    
    return X
    
def load_data(dataset):

    array = dataset.values
    X = array[:,:-1]
    Y = array[:,-1]
    for i in range(len(Y)):
        if Y[i] == 'Iris-setosa' or Y[i] == 'NO':
            Y[i] = -1
        else:
            Y[i] = 1
    
    return X, Y

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
                    g.scatter(gr[i][m][:,0], gr[i][m][:,1], color = colors_gr[m])
                    g.scatter(gr_train[i][m][:,0], gr_train[i][m][:,1], marker = '.', color = colors_pts[m])
                    #if pts[i][m][:,:] != []:
                    g.scatter(pts[i][m][:,0], pts[i][m][:,1], marker = markers_pts[m], color = colors_pts[m])
                i = i + 1    
    plt.show()

def graphic(model):
    train = model[0]
    test = model[1]
    gamma = model[2]
    '''parameters = model[2]
    mean = parameters[0]
    m_covariance = parameters[1]
    ps_priori = parameters[2]'''

    new_trains = []
    new_tests = []
    qtd_attributs = len(train[0,:-1])
    for i in range(qtd_attributs):
        for j in range(i+1,qtd_attributs):
            new_train = []
            new_test = []
            X = train[:,i]
            Y = train[:,j]
            
            X_t = test[:,i]
            Y_t = test[:,j]

            for c in range(len(train)):
                new_train.append(np.array([X[c],Y[c], train[c][-1]]))
            
            for t in range(len(test)):
                new_test.append(np.array([X_t[t],Y_t[t], test[t][-1]]))
            
            new_trains.append(new_train)
            new_tests.append(new_test)

    new_trains = np.array(new_trains)
    new_tests = np.array(new_tests)

    pts = []
    colors_pts = ['darkred','darkblue','darkgreen']
    colors_pts_ = []
    markers_pts = ['*','+','x']
    markers_pts_ = []
    bayesian = BaysianClassifierWithRejection()
    for i in range(len(new_tests)):
        bayesian.fit(new_trains[i], gamma)
        pred = bayesian.prediction(new_tests[i])
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        new_tests[i] = np.concatenate((new_tests[i][:,:-1],pred), axis = 1)
        pts_pontos = []
        for c in classes:
            points = new_tests[i][np.where(new_tests[i][:,-1] == c)]
            pts_pontos.append(points)
        pts.append(pts_pontos)

    for c in range(len(classes)):
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
    for train in new_trains:
        bayesian.fit(train, gamma)
        pred = bayesian.prediction(data_test_all)
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
    
    for c in range(len(classes)):
        colors_gr_.append(colors_gr[c])

    n_row, n_column = 1, 1
    if len(gr) == 6:
        n_row, n_column = 2, 3
    elif len(gr) == 15:
        n_row, n_column = 3, 5
    elif len(gr) == 45:
        n_row, n_column = 5, 7

    #graph(n_row, n_column, pts, gr, colors_pts_, colors_gr_, markers_pts_)
    graph(n_row, n_column, pts, gr, gr_train, colors_pts_, colors_gr_, markers_pts_)


if __name__ == "__main__":

    entered = raw_input('Choose the base if you want run the KNN:\n(a)-Iris;\n(b)-Vertebral Column;\n(c)-Base Artificial.\n')

    if(entered == 'a'):    
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pd.read_csv('iris.csv', names = names)
        X, Y = load_data(dataset)
    elif(entered == 'b'):
        dataset = pd.read_table('column_3C.dat', sep = " ")
        X, Y = load_data(dataset)
    else:
        dataset = pd.read_csv('artificial_3.csv', sep = " ")
        array = dataset.values
        X = array[:,:-1]
        Y = array[:,-1]

    X = normalization(X)

    Y = np.reshape(Y, (len(Y),1))
    dataset = np.concatenate((X,Y), axis = 1)

    baysianClassifier = BaysianClassifierWithRejection(kernel = 'linear')

    realizations = 20

    accuracies_test = []
    accuracies_train = []
    models = []
   
    best_gammas = []
    mat_conf_test = []
    mat_conf_train = []
    for realization in range(realizations):
        print(realization)
        gammas = [x for x in range(1, 10)]  
        train, test = train_test_split(dataset, 0.2)
        accuracies_gamma_train = []
        accuracies_gamma = []
        matrices_gamma = []
        conf_train = []
        conf_t = []
        for gamma in gammas:
            total = baysianClassifier.fit(train, gamma)

            #Predictions in training set 
            predictions_train = baysianClassifier.prediction(train)
            reals_train = [y for y in train[:,-1]]
            conf_matrix_train, acc = baysianClassifier.accuracy(reals_train, predictions_train, len(set(train[:,-1])))
            accuracies_gamma_train.append(acc)
            conf_train.append(conf_matrix_train)
            #Predictions in test set
            predictions_test = baysianClassifier.prediction(test)
            reals_test = [y for y in test[:,-1]]
            conf_matrix_t, acc = baysianClassifier.accuracy(reals_test, predictions_test, len(set(train[:,-1])))
            print(conf_matrix_t)
            print(acc)
            conf_t.append(conf_matrix_t)
        
            accuracies_gamma.append(acc)
            matrices_gamma.append(conf_matrix_t)
        
        max_accuracy = max(accuracies_gamma)
        index_max = accuracies_gamma.index(max_accuracy)

        accuracies_train.append(accuracies_gamma_train[index_max])
        accuracies_test.append(accuracies_gamma[index_max])
        mat_conf_test.append(conf_t[index_max])
        mat_conf_train.append(conf_train[index_max])
        gamma_best = gammas[index_max]

        best_gammas.append(gamma_best)
        models.append((train, test, gamma_best))
    
    print(best_gammas)

    mean_accuracy = np.mean(accuracies_test)
    diff = [abs(x - mean_accuracy) for x in accuracies_test]
    best = min(diff)
    best_index = diff.index(best)

    print("######## Train ########")
    print("Media das acuracias: ", np.mean(accuracies_train))
    print("Desvio padrao : ", np.std(accuracies_train))
    #print("Tempo medio: ", np.mean(times_train))
    print(" ")
    print("######## Teste ########")
    print("Media das acuracias: ", mean_accuracy)
    print("Desvio padrao : ", np.std(accuracies_test))
    #print("Tempo medio: ", np.mean(times_test))
    print(" ")
    print("Realizacao escolhida: ", best_index+1) 
    print("Melhor Matriz de confusao:")
    print(mat_conf_test[best_index])
    print("Melhor Acuracia:")
    print(accuracies_test[best_index])

    graphic(models[best_index])
        