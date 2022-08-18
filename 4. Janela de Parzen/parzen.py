import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import math
import matplotlib.pyplot as plt
import time

class BaysianClassifier:

    def __init__(self):
        pass
    
    def fit(self, train):
        #self.train_set = train[:]
        classes = set(train[:,-1])
        self.classes = classes
        tuples = []
        for c in classes:
            train_ = train[np.where(train[:,-1] == c)]
            prob_ = float(len(train_))/len(train)
            tuples.append((prob_, train_))
         
         
        self.tuples = tuples     
        return tuples 

    def likelihood(self, x, c, l, h):
        c = int(c)
        #sigmai = self.tuples[c][0]
        xi = self.tuples[c][1][:,:-1]
        N = len(xi)
        #mi = self.tuples[c][1]
        x = np.array(x)
        #print "x: ", x.shape
        xi = np.array(xi)
        #print "xi: ", xi.shape
    
        #print "N: ", N
        #dot = np.array(np.dot((x - xi).T, (x - xi)), dtype = np.float32)
        #print (-dot)/(2.0*pow(h, 2))
        Px = []
        for x_i in xi:
            dot = np.array(np.dot((x - x_i).T, (x - x_i)), dtype = np.float32)
            Px.append((1.0/(pow(2*math.pi, float(l)/2) * pow(h, l) )) * np.exp((-dot)/(2.0*pow(h, 2))))
        #print Px
        Px = np.mean(Px)
        #P = np.mean((1.0/(pow(2*math.pi, float(l)/2) * pow(h, l) )) * np.exp((-dot)/(2.0*pow(h, 2))))
        return Px

    def predict(self, x, h):
        posteriori_probs = []
        l = len(x)
        #print "tamanho ", l 
        for c in self.classes:
            c = int(c)
            priori = self.tuples[c][0]
            posteriori = self.likelihood(x, c, l, h) * priori
            posteriori_probs.append(posteriori)
        
        #print posteriori_probs
        p = max(posteriori_probs)
        c = posteriori_probs.index(p)
        return c

    def prediction(self, test, h):
        pred = []
        for x in test[:,:-1]:
            pred.append(self.predict(x, h))

        return pred

    def accuracy(self, actual, predicted, n_classes):
        correct = 0
        confusion_matrix = np.zeros((n_classes,n_classes), int)    
        for i in range(len(actual)):
            actual[i] = int(actual[i])
            confusion_matrix[actual[i]][predicted[i]] = confusion_matrix[actual[i]][predicted[i]] + 1
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
    #print qtd_total_data
    qtd_test = int(test_size*qtd_total_data)
    #print qtd_test
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
        if Y[i] == 'Iris-setosa' or Y[i] == 'DH' or Y[i] == 2:
            Y[i] = 0
        elif Y[i] == 'Iris-versicolor' or Y[i] == 'SL' or Y[i] == 4:
            Y[i] = 1
        else:
            Y[i] = 2

    return X, Y

def load(dataset):

    array = dataset.values
    array = array[np.where(array[:,-2] != '?')]
    array = array.astype('float')

    X = array[:,:-1]
    Y = array[:,-1]
    for i in range(len(Y)):
        Y[i] = Y[i]-1

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
                    #print m
                    #print gr_train[i][m][:,0], gr_train[i][m][:,1], colors_pts[m]
                    g.scatter(gr[i][m][:,0], gr[i][m][:,1], color = colors_gr[m])
                    g.scatter(gr_train[i][m][:,0], gr_train[i][m][:,1], marker = '.', color = colors_pts[m])
                    if pts[i][m][:,:] != []:
                        g.scatter(pts[i][m][:,0], pts[i][m][:,1], marker = markers_pts[m], color = colors_pts[m])
                i = i + 1    
    plt.show()

def graphic(model):
    train = model[0]
    test = model[1]
    h = model[2]
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
    bayesian = BaysianClassifier()
    for i in range(len(new_tests)):
        bayesian.fit(new_trains[i])
        pred = bayesian.prediction(new_tests[i], h)
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
        bayesian.fit(train)
        pred = bayesian.prediction(data_test_all, h)
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

    entered = input('Choose the base if you want run the B.C. parzen:\n(a)-Iris;\n(b)-Vertebral Column;\n(c)-Base Artificial;\n(d)-Dermatology;\n(e)-Breast Canser.\n')

    if(entered == 'a'):    
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pd.read_csv('iris.csv', names = names)
        X, Y = load_data(dataset)
        X = normalization(X)
    elif(entered == 'b'):
        dataset = pd.read_table('column_3C.dat', sep = " ")
        X, Y = load_data(dataset)
        X = normalization(X)
    elif(entered == 'c'):
        dataset = pd.read_csv('base_artificial2.csv', sep = " ")
        array = dataset.values
        array = array[:,1:]
        X = array[:,:-1]
        Y = array[:,-1]
        X = normalization(X)

    elif(entered == 'd'):
        dataset = pd.read_table('dermatology.data', sep = ",")

        X, Y = load(dataset)
    
        X = normalization(X)
    else:
        dataset = pd.read_table('breast_cancer.data', sep = ",")
        X, Y = load_data(dataset)

        Y = np.reshape(Y, (len(Y),1))
        dataset = np.concatenate((X,Y), axis = 1)

        dataset = dataset[np.where(X[:,-4] != '?')]
            
        dataset = dataset.astype('float')
        
        X = dataset[:,:-1]
        Y = dataset[:,-1]

        X = normalization(X)
        
    


    #X = normalization(X)

    Y = np.reshape(Y, (len(Y),1))
    dataset = np.concatenate((X,Y), axis = 1)

    baysianClassifier = BaysianClassifier()

    realizations = 20

    accuracies = []
    accuracies_train = []
    conf = []
    conf_trains = []
    models = []
    metrics_avaluation = []
    times_train = []
    times_test = []
    for realization in range(realizations):
        train, test = train_test_split(dataset, 0.2)
        start_train = time.time()
        parameters = baysianClassifier.fit(train)
        end_train = time.time()

        times_train.append(end_train - start_train)
        h_ = [x/10.0 for x in range(1,10)]
        accs_train = []
        for h in h_:
        ################ Treinamento ################
            predictions_train = baysianClassifier.prediction(train, h)
            reals_train = [y for y in train[:,-1]]
            #print("tamanho ", len(set(train[:,-1])))
            
            conf_matrix_train, accuracy_train = baysianClassifier.accuracy(reals_train, predictions_train, len(set(train[:,-1])))
            conf_trains.append(conf_matrix_train)
            #print(accuracy_train)
            accs_train.append(accuracy_train)
        
        index_best = accs_train.index(max(accs_train))
        accuracies_train.append(accs_train[index_best])
        h_best = h_[index_best]
        
        
        ################ Teste ################
        start_test = time.time()
        predictions = baysianClassifier.prediction(test, h_best)
        end_test = time.time()

        times_test.append(end_test - start_test)
        reals = [y for y in test[:,-1]]
        #print("tamanho ", len(set(train[:,-1])))
        conf_matrix, accuracy = baysianClassifier.accuracy(reals, predictions, len(set(train[:,-1])))
        conf.append(conf_matrix)
        print("Confusion Matrix: ")
        print conf_matrix
        print("Accuracy: ", accuracy)
        accuracies.append(accuracy)
        models.append((train, test, h_best))
        metrics_avaluation.append(accuracy)

    #index_max_accuracy_model = metrics_avaluation.index(max(metrics_avaluation))
    #graphic(models[index_max_accuracy_model])

    mean_accuracy = np.mean(accuracies)
    diff = [abs(x - mean_accuracy) for x in accuracies]
    best = min(diff)
    best_index = diff.index(best)

    print("######## Train ########")
    print("Media das acuracias: ", np.mean(accuracies_train))
    print("Desvio padrao : ", np.std(accuracies_train))
    print("Tempo medio: ", np.mean(times_train))
    print(" ")
    print("######## Teste ########")
    print("Media das acuracias: ", mean_accuracy)
    print("Desvio padrao : ", np.std(accuracies))
    print("Tempo medio: ", np.mean(times_test))
    print(" ")
    print("Realizacao escolhida: ", best_index+1) 
    print("Melhor Matriz de confusao:")
    print(conf[best_index])
    print("Melhor Acuracia:")
    print(accuracies[best_index])

    graphic(models[best_index])