import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import math
import matplotlib.pyplot as plt

class BaysianClassifierWithRejection:

    def __init__(self):
        pass
    
    def fit(self, train):
        classes = set(train[:,-1])
        self.classes = classes
        tuples = []
        for c in classes:
            train_ = train[np.where(train[:,-1] == c)]
        
            cov_ = np.cov(train_[:,:-1].astype(float).T)
            mean_ = np.mean(train_[:,:-1], axis = 0)
            prob_ = float(len(train_))/len(train)
            tuples.append((cov_, mean_, prob_))
         
         
        self.tuples = tuples     
        return tuples 

    def likelihood(self, x, c, l):
        c = int(c)
        sigmai = self.tuples[c][0]
        mi = self.tuples[c][1]
        P = (1.0/(pow(2*math.pi, float(l)/2) * pow(det(sigmai), 1.0/2) )) * math.exp(-1.0/2 * np.dot(np.dot((x - mi).T, inv(sigmai)), x - mi))
        return P 

    def predict(self, x, threshold):
        #print "Thres: ", threshold
        posteriori_probs = []
        l = len(x)
      
        px = []
        evidences = []
        for c in self.classes:
            c = int(c)
            likelihood = self.likelihood(x,c,l)
            px.append(likelihood)
            evidences.append(likelihood*self.tuples[c][2])

        for c in self.classes:
            c = int(c)
            priori = self.tuples[c][2]
            posteriori = (px[c] * priori)/sum(evidences) 
            posteriori_probs.append(posteriori)
        
        p = max(posteriori_probs)
        #print sum(posteriori_probs)
        #print p, posteriori_probs
        #if p < (0):
        #if p < (1 - threshold):
        #print "Prob: ", p
        #print "Limite inferior: ", (0.5 - threshold)
        #print "Limite superior: ", (0.5 + threshold)
        #print "threshold: ", threshold
        if p > (0.5 - threshold)  and p < (0.5 + threshold):
            #print "Rejeitado"
            c = 2
        else:
            c = posteriori_probs.index(p)
        
        #print "Classe: ", c

        return c

    def prediction(self, test, threshold):
        pred = []
        for x in test[:,:-1]:
            pred.append(self.predict(x, threshold))

        return pred

    def accuracy(self, actual, predicted, n_classes):
        correct = 0
        rejecteds = 0
        total = len(actual)
        confusion_matrix = np.zeros((n_classes,n_classes), int)    
        for i in range(total):
            actual[i] = int(actual[i])
            confusion_matrix[actual[i]][predicted[i]] = confusion_matrix[actual[i]][predicted[i]] + 1
            if predicted[i] == 2:
                rejecteds = rejecteds + 1

            else:
                if actual[i] == predicted[i]:
                    correct += 1
        
        #if rejecteds < total:
        acc = correct/ float(total - rejecteds) * 100.0
        #else:
        #    acc = 0.0
        rej = rejecteds/float(total)
        err = 100 - acc 
        #acc = correct / float(len(actual)) * 100.0
        return confusion_matrix, acc, rej, err, rejecteds


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
        if Y[i] == 'Iris-setosa' or Y[i] == 'NO':
            Y[i] = 0
        else:
            Y[i] = 1
    #print Y
    return X, Y

def graph(n_row, n_column, pts, gr, gr_train, colors_pts, colors_gr, markers_pts):
    #print "coors: ", colors_pts
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
                print "colors: ", colors_pts
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
    threshold = model[2]
    #print "t: ", t
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
        bayesian.fit(new_trains[i])
        pred = bayesian.prediction(new_tests[i], threshold)
        #print pred
        classes = set(pred)
        pred = np.reshape(pred, (len(pred),1))
        new_tests[i] = np.concatenate((new_tests[i][:,:-1],pred), axis = 1)
        pts_pontos = []
        for c in classes:
            points = new_tests[i][np.where(new_tests[i][:,-1] == c)]
            pts_pontos.append(points)
        pts.append(pts_pontos)

    #print classes
    for c in range(len(classes)):
        #print c
        colors_pts_.append(colors_pts[c])
        markers_pts_.append(markers_pts[c])
    #print "color_pts: ", colors_pts_
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
    colors_gr = ['salmon', 'skyblue', 'white', 'lightgreen']
    colors_gr_ = []
    tests_all = []
    for train in new_trains:
        bayesian.fit(train)

        pred = bayesian.prediction(data_test_all, threshold)
        #print "predicao: ", pred
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
    #print classes
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

    entered = input('Choose the base if you want run the KNN:\n(a)-Iris;\n(b)-Vertebral Column;\n(c)-Base Artificial.\n')

    if(entered == 'a'):    
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pd.read_csv('iris.csv', names = names)
        X, Y = load_data(dataset)
    elif(entered == 'b'):
        dataset = pd.read_table('column_3C.dat', sep = " ")
        X, Y = load_data(dataset)
    else:
        dataset = pd.read_csv('artificial_1.csv', sep = " ")
        array = dataset.values
        X = array[:,:-1]
        Y = array[:,-1]

    X = normalization(X)

    Y = np.reshape(Y, (len(Y),1))
    dataset = np.concatenate((X,Y), axis = 1)

    baysianClassifier = BaysianClassifierWithRejection()

    realizations = 20

    accuracies_test = []
    rejections_test = []
    models = []
    #thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
    WR = [0.04, 0.12, 0.24, 0.36, 0.48]
    best_thresholds = [] 
    bests = []
    thres = []
    for realization in range(realizations):
        print(realization)
        train, test = train_test_split(dataset, 0.2)
        baysianClassifier.fit(train)
        accuracies = []
        rejections = []

        accuracies_t = []
        rejections_t = [] 
            
        for i in range(len(WR)):
            risks =  []
            accuracies_train = []
            rejections_train = [] 
            thresholds = [t/20.0 for t in range(0,11)]   
            min_risks = []
            for t in thresholds:
                #t = t/20.0
                predictions = baysianClassifier.prediction(train, t)
                reals = [y for y in train[:,-1]]
                conf_matrix, accuracy, rejection, errors, rejecteds = baysianClassifier.accuracy(reals, predictions, len(set(train[:,-1])) + 1)
        
                er = errors/100.0
                risk = er + WR[i] * rejection
                risks.append(risk)
                #print("Risco Minimo: ", risk)
                accuracies_train.append(accuracy)
                rejections_train.append(rejection)
                
            #print("riscos: ", risks)
            min_risk = min(risks)
            min_risks.append(min_risk)
            best_threshold = risks.index(min_risk)
            print("T: ", thresholds[best_threshold])
            best_thresholds.append(thresholds[best_threshold])
            accuracies.append(accuracies_train[best_threshold])
            rejections.append(rejections_train[best_threshold])

            predictions_test = baysianClassifier.prediction(test, thresholds[best_threshold])
            reals_test = [y for y in test[:,-1]]
            conf_matrix_t, accuracy_t, rejection_t, errors_t, rejecteds = baysianClassifier.accuracy(reals_test, predictions_test, len(set(train[:,-1])) + 1)
            #print("numero de rejeitados - teste:", rejecteds)
            
            print("Confusion Matrix - test: ")
            print(conf_matrix_t)
            print("Accuracy - test: ", accuracy_t)
            print("Rejection - test: ", rejection_t)
            print("Errors - test: ", errors_t)
            
            risk_t = errors_t + thresholds[i] * rejection_t
            #print "Risco Minimo - test: ", risk_t
            accuracies_t.append(accuracy_t)
            rejections_t.append(rejection_t)

        best = min(min_risks)
        index = min_risks.index(best)
        accuracies_test.append(accuracies_t)
        rejections_test.append(rejections_t)
        print("t:", thresholds[index])
        print(best_thresholds)
        models.append((train, test, thresholds[best_threshold]))
        thres.append(thresholds[index])
        bests.append(best)


    print("rejeicao")
    print(rejections_test)
    print("acuracies: ", accuracies_test)
    print("Rejeicao: ", np.mean(rejections_test, axis = 0))
    print("Taxa de acerto: ", np.mean(accuracies_test, axis = 0)) 
    plt.plot(np.mean(rejections_test, axis = 0), np.mean(accuracies_test, axis = 0), color = 'red')
    plt.xlabel('Taxa de Rejeicao')
    plt.ylabel('Taxa de Acerto')
    plt.show()
        
    print("################## Treinamento ##################")
    print("Media das acuracias: ", np.mean(accuracies))
    print("Media da taxa de rejeicao: ", np.mean(rejections))
    print("Desvio padrao das acuracias: ", np.std(accuracies))
    print("Desvio padrao das taxas de rejeicao  - test: ", np.std(rejections))
    print(" ")
    print("################## Teste ##################")
    print("Media das acuracias: ", np.mean(accuracies_t))
    print("Media da taxa de rejeicao: ", np.mean(rejections_t))
    print("Desvio padrao das acuracias  - test: ", np.std(accuracies_t))
    print("Desvio padrao das taxas de rejeicao  - test: ", np.std(rejections_t))

    best_index = bests.index(min(bests))
    
    graphic(models[best_index])