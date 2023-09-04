import codecademylib3_seaborn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# step 1 - load the dataset.
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

# step 2 - what does the data look like?
# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)

# step 3 - what does the target look like
# print(breast_cancer_data.target) 
# print(breast_cancer_data.target_names)

# step 4 - 6 - create training and validation sets

training_set, validation_set, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# step 7 - confirm previous steps worked correctly. 
# print(len(training_set))
# print(len(training_labels))

# step 8 - 9 - run classifier.
classifier = KNeighborsClassifier(n_neighbors = 3)

# step 10 - train classifier.
classifier.fit(training_set, training_labels)

# step 11 - test accuracy.
accuracy = classifier.score(validation_set, validation_labels)
# print(accuracy)

# step 12 - print a range of ks

accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_set, training_labels)
  accuracy = classifier.score(validation_set, validation_labels)
  accuracies.append(accuracy)

# step 13 - 17 - plot the k to determine the best k value to use.
k_list = list(range(1, 101))

plt.plot(k_list, accuracies)
plt.title('Breast Cancer Classifier Accuracy')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.show()
plt.clf()
