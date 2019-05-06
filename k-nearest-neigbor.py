import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd 

accurancies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.csv')
    df.replace('?', -99999, inplace=True)

    # remove useless data
    '''
        if you comment this out it wreaks havoc on your accuracy.  
        Which goes to show you that you need to use good data.
    '''

    df.drop(['id'], 1, inplace=True)

    # Features
    X = np.array(df.drop(['class'], 1))
    # Labels
    y = np.array(df['class'])

    # Shuffle data and seperate into training and test data 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    # #print(accuracy)

    # example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)

    # prediction = clf.predict(example_measures)
    # #print(prediction)

    accurancies.append(accuracy)

print(sum(accurancies)/len(accurancies))