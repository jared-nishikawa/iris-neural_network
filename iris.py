#!/usr/bin/python

import numpy as np
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def categorical(label):
    try:
        values = [0]*3
        values[labels.index(label)] = 1
        return values
    except:
        exit("Unknown label %s" % label)

def get_label(values):
    V = list(values)
    M = max(V)
    return labels[V.index(M)]

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    df = pd.read_csv('iris.data')
    rows = [row[1].tolist() for row in df.iterrows()]
    random.shuffle(rows)

    # Randomly set training data and test data
    training_data = rows[:100]
    test_data = rows[100:]

    X = [row[:4] for row in training_data]
    Y = [categorical(row[-1]) for row in training_data]

    X = np.array(X)
    Y = np.array(Y)

    # Create model
    print "Creating model..."
    model = Sequential()

    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    # Compile model
    print "Compiling model..."
    model.compile(loss='binary_crossentropy',
            optimizer='adam', 
            metrics=['accuracy'])

    # Fit model
    print "Fitting model..."
    model.fit(X, Y, epochs=1000, batch_size=10)

    # Evaluate model
    print "Evaluating model..."
    scores = model.evaluate(X,Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # Save model
    model.save('model.h5')

    # Test model
    X_test = [row[:4] for row in test_data]
    X_test = np.array(X_test)

    Y_test = [row[-1] for row in test_data]

    # Predictions
    Z = model.predict(X_test, batch_size=8, verbose=1)
    print "Z", len(Z)
    correct = 0
    total = 0
    for i,z in enumerate(Z):
        total += 1
        label = get_label(z)
        print "Predicted:", label
        print "Observed:", Y_test[i]
        if label == Y_test[i]:
            correct += 1

    print "Accuracy:", "%.2f" % (float(correct)*100/total)


