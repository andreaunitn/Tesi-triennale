import numpy

def MSE(image_1, image_2):

    # altezza e larghezza immagini
    height_1 = image_1.shape[0]
    width_1 = image_1.shape[1]

    height_2 = image_2.shape[0]
    width_2 = image_2.shape[1]

    # controllo risoluzione
    if not (height_1 == height_2 and width_1 == width_2):
        return
    
    # calcolo mse
    mse = numpy.mean((image_1 - image_2) ** 2)

    return mse

from math import log10, sqrt

def PSNR(mse):

    # massimo valore dei pixel
    max_pixel = 255

    # controllo valore mse
    if mse != 0:

        # calcolo psnr
        p = 20 * log10(max_pixel / sqrt(mse))
        return p
    else:
        return numpy.NaN


from sklearn import manifold

def DimensionalityReduction(feature):

    # inizializzazione isomap
    isomap = manifold.Isomap(n_components = 2)
    isomap.fit(feature)
    
    # trasformazione feature
    feature_transformed = isomap.transform(feature)

    return feature_transformed

import matplotlib.pyplot as plt

def FeatureVisualization(feature_transformed):

    # separazione delle feature nelle singole classi
    web_mac = feature_transformed[0:900]
    app_win = feature_transformed[900:1800]
    web_ipad = feature_transformed[1800:2700]
    app_mac = feature_transformed[2700:3600]
    iphone = feature_transformed[3600:4500]
    web_win = feature_transformed[4500:5400]

    # visualizzazione feature
    # inizializzazione grafico
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # inserimento dati + legenda
    ax.scatter(web_mac[:, 0], web_mac[:, 1], label = "WEB-MAC")
    ax.scatter(app_win[:, 0], app_win[:, 1], label = "APP-WIN")
    ax.scatter(web_ipad[:, 0], web_ipad[:, 1], label = "WEB-IPAD")
    ax.scatter(app_mac[:, 0], app_mac[:, 1], label = "APP-MAC")
    ax.scatter(iphone[:, 0], iphone[:, 1], label = "IPHONE")
    ax.scatter(web_win[:, 0], web_win[:, 1], label = "WEB-WIN")
    ax.legend()

    # salvataggio grafico
    plt.savefig("path_to_folder/file.pdf")

    plt.show()


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def Classification(train_set_x, train_set_y, test_set_x, test_set_y, mod):

    # classificazione
    if mod == "svm":

        # support vector machine
        clf = svm.SVC(kernel = "rbf", C = 1, gamma = 0.1)
    elif mod == "rf":

        # random forest
        clf = RandomForestClassifier(n_estimators = 100)

    # training
    clf.fit(train_set_x, train_set_y)

    # testing
    predictions = clf.predict(test_set_x)

    # costruzione confusion matrix
    cm = confusion_matrix(test_set_y, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
