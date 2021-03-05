import numpy as np
import random
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from numpy.random import choice
from skmultiflow.trees import HoeffdingTreeClassifier
import sklearn.metrics as metrics
from IOE import IOE_Classifier
from sklearn.metrics import accuracy_score



def experiment(data, classes, threshold=0.05, forgetting_factor=0.9,m=10,warm_up =1000,window=1):
    random.seed(0)
    np.random.seed(0)

    classes_precision = {}
    classes_recall = {}
    recalls_for_all_classes = {}
    classes_recall_forgetting_factor = {}
    for value in classes:
        classes_precision[int(value)] = []
        recalls_for_all_classes[int(value)] = []
        recalls_for_all_classes[int(value)].append(0)
        classes_precision[int(value)].append(0)
        classes_precision[int(value)].append(0)

    for value in classes:
        classes_recall[int(value)] = []
        classes_recall[int(value)].append(0)
        classes_recall[int(value)].append(0)
    for value in classes:
        classes_recall_forgetting_factor[int(value)] = 0
    measure = ClassificationMeasurements()
    clf = IOE_Classifier(HoeffdingTreeClassifier(), threshold=threshold, forgetting_factor=forgetting_factor, m=m)
    i = 0
    Xtmp = data[i:i + warm_up]
    X = []
    y = []
    for var in Xtmp:
            X.append(var[:-1])
            y.append(int(var[-1]))
    X = np.array(X)
    y = np.array(y)
    clf.partial_fit(X, y, classes, warm_up=True)
    i += warm_up
    all_x = 0
    predictions = []
    ys = []

    while (i + window < len(data)):
        j = i + window
        Xtmp = data[i:j]
        y_labels = data[i:j,-1]
        X = []
        y = []

        for (count, var) in enumerate(Xtmp):
            # print(var)
            result = clf.predict(np.array([var[:-1]]))
            result = result[0]
            predictions.append(result)
            ys.append(y_labels[count])

            measure.add_result(y_true=y_labels[count], y_pred=result, weight=1.0)
            if (y_labels[count] == result):
                classes_precision[result][0] += 1
                classes_recall[result][0] += 1
            else:
                if result in classes:
                    classes_precision[result][1] += 1
                else:
                    print(f"{result} not in classes")
                classes_recall[y_labels[count]][1] += 1
            for key, rec in classes_recall.items():
                if (rec[0] + rec[1]) != 0:
                    recalls_for_all_classes[key].append(rec[0] / (rec[0] + rec[1]))
                else:
                    recalls_for_all_classes[key].append(0)

                X.append(var[:-1])
                y.append(int(var[-1]))

        X = np.array(X)
        y = np.array(y)
        clf.partial_fit(X, y, classes=classes)
        i = j
        print(f"{i} out of {len(data)}", end="\r")


    Final_result = []
    Final_result.append(measure.get_accuracy())
    Final_result.append(measure.get_kappa())
    Final_result.append(measure.get_kappa_m())
    Final_result.append(measure.get_kappa_t())
    Final_result.append(classes_recall.items())
    print(f"Finished")
    print(f"Final Acc is {measure.get_accuracy()}")
    print(f"scikit lear acc score {accuracy_score(predictions,ys)}")
    print(f"Final Kappa is {measure.get_kappa()}")
    print(f"Final Kappa_M is {measure.get_kappa_m()}")
    print(f"Final Kappa_T is {measure.get_kappa_t()}")
    print(f"Recall is {measure.get_recall()}")
    print(f"Precision is {measure.get_precision()}")
    recall = 1
    recalls = []
    precisions = []
    macro_recall = 0
    macro_precision = 0

    for key, var in classes_recall.items():
        if (var[0] + var[1]) != 0:
            recall *= (var[0] / (var[0] + var[1]))
            print(f"class {str(key)} recall : {str(var[0] / (var[0] + var[1]))} ")
            print(var[0] + var[1])
            recalls.append((var[0] / (var[0] + var[1])))
            macro_recall += (var[0] / (var[0] + var[1]))
    print(f"macro recall is {macro_recall / len(classes)}")
    for key, var in classes_precision.items():
        #         recall*=(var[0]/( var[0]+var[1]))
        if (var[0] + var[1]) != 0:
            print(f"class {str(key)} precision : {str(var[0] / (var[0] + var[1]))} ")
            macro_precision += (var[0] / (var[0] + var[1]))
            precisions.append((var[0] / (var[0] + var[1])))
        else:
            precisions.append(0)
    print(f"macro precision is {macro_precision / len(classes)}")
    macro_f1 = 0
    for i in range(len(recalls)):
        if precisions[i] + recalls[i] != 0:
            macro_f1 += 2 * recalls[i] * precisions[i] / (precisions[i] + recalls[i])
    print(f"macro_f1 is {macro_f1 / len(recalls)}")

    Final_result.append(recalls)
    Final_result.append(recalls_for_all_classes)


    print(f"G_mean {recall ** (1 / len(recalls))}")
    Final_result.append(recall ** (1 / len(recalls)))


    return Final_result

