import numpy as np
import pandas as pd

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn import svm
from sklearn import metrics

import matplotlib.pyplot as plt

import pickle

# Constants
highest_K = 12
linear_retrains = 30
KNN_test_size = 500
#total_best_k = 3

# 1 Logistic Regression: 1.1 is model training and 1.2 is reusing the saved model
# 2 KNN: 2.1 is calculating total_best_k and 2.2 if you already know total_best_k
# 3 SVM
method = 1.1

# Load data
df = pd.read_csv("train.csv")

# Pick only data relevant to the task
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Cleaning data (replacing missing values with averages)
average_age = df["Age"].mean()
average_fare = df["Fare"].mean()
df["Age"] = df["Age"].fillna(average_age)

# Prepocessing, converting strings to ints
encoder = preprocessing.LabelEncoder()

sex = encoder.fit_transform(list(df["Sex"]))
df["Sex"] = sex
# Male = 1, female = 0

# Split data into X and Y
label = "Survived"

X = np.array(df.drop([label], axis=1))
Y = np.array(df[label])

# Linear regression model
best = 0
model = None
if round(method, 0) == 1:
    print("Model used is Logistic Regression")
    #plt.scatter(df.Fare, df.Survived)
    #plt.show()
    if method == 1.1:
        for i in range(linear_retrains):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
            linear = linear_model.LogisticRegression(max_iter=10000, C=2)
            linear.fit(x_train, y_train)
            #print(linear)
            acc = linear.score(x_test, y_test)
            #print(acc)
            if acc > best:
                best = acc
                f = open("titanic.pickle", "wb")
                pickle.dump(linear, f)
        print(f"Best acc is {best}")

    pickle_in = open("titanic.pickle", "rb")
    model = pickle.load(pickle_in)

# KNN model
elif round(method, 0) == 2:
    print(f"Model used is KNN")
    if method == 2.1:
        freq = dict()
        for i in range(1, highest_K):
            freq[i] = 0

        for j in range(KNN_test_size):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

            best_acc = 0
            best_k = 0
            for i in range(1, highest_K):
                model = KNeighborsClassifier(i)
                model.fit(x_train, y_train)
                acc = model.score(x_test, y_test)
                if acc > best_acc:
                    best_acc = acc
                    best_k = i
            freq[best_k] += 1
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        #print(freq)
        total_best_k = freq[0][0]
        #print(total_best_k)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    model = KNeighborsClassifier(total_best_k)
    model.fit(x_train, y_train)

# SVM model
elif method == 3:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2)
    model = svm.SVC(kernel="linear", C=2)
    model.fit(x_train, y_train)

    predict = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, predict)
    print(acc)

else:
    print("No method given. Exiting...")
    exit(1)

# Loading test data
test_df = pd.read_csv("test.csv")
test_df = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Preprocessing test data: converting string (sex) to integer
test_sex = encoder.fit_transform(list(test_df["Sex"]))
test_df["Sex"] = test_sex

# Cleaning test data (replacing missing values with averages)
test_df["Age"] = test_df["Age"].fillna(average_age)
test_df["Fare"] = test_df["Fare"].fillna(average_fare)

test_X = np.array(test_df)

# Making predictions
solutions = model.predict(test_X)
for i in range(len(solutions)):
    solutions[i] = abs(round(solutions[i], 0))
print(solutions)

# Writing solutions to 'solution.csv' file
f = open("solution.csv", "w")
f.write("PassengerId,Survived")
f.close()
f = open("solution.csv", "a")
for i in range(len(solutions)):
    f.write(f"\n{892+i},{int(solutions[i])}")
f.close()