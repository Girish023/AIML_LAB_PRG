import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn import metrics

# Read dataset to pandas dataframe
dataset = pd.read_csv("C:\\Users\\GIRISH CHANDRA\\7th SEM AI&ML\\AI&ML LAB\\Iris.csv", header=0)
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.1) 

classifier = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

print("\n-------------------------------------------------------------------------")
print('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print("-------------------------------------------------------------------------")
for original, predicted in zip(y_test, y_pred):
    correctness = 'Correct' if original == predicted else 'Wrong'
    print('%-25s %-25s %-25s' % (original, predicted, correctness))
print("-------------------------------------------------------------------------")

print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))  
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred)) 
print('Accuracy of the classifier is %0.2f' % metrics.accuracy_score(y_test, y_pred))