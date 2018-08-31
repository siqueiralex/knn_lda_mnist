import os
import numpy as np
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 

def print_report(predictions, targets):
    confusion_matrix = np.zeros((10, 10), dtype=np.int)
    for i in range(len(predictions)):
        confusion_matrix[targets[i]][predictions[i]]+=1
    print(confusion_matrix)    
    print(classification_report(targets, predictions)) 

path = os.path.join("datasets", "mnist")
mndata = MNIST(path)
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

#sample tests
test_images = test_images[:1000]
test_labels = test_labels.tolist()[:1000]

k = 5
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
knn.fit(images, labels.tolist())
predictions = knn.predict(test_images)

print("KNN k=10")
print_report(predictions, test_labels)
print()

knn.n_neighbors=10
predictions = knn.predict(test_images)
print("KNN k=100")
print_report(predictions, test_labels)    
print()


knn.n_neighbors=244
predictions = knn.predict(test_images)
print("KNN k=244 -> sqrt(60k)")
print_report(predictions, test_labels)    
print()

lda = LinearDiscriminantAnalysis()
lda.fit(images, labels)

predictions = lda.predict(test_images)
print("LDA")
print_report(predictions, test_labels)
print()