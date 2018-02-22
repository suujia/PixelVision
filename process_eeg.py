from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_features_and_labels(test_size):
    data, metadata = arff.loadarff(open('data/EEG-Eye-State.arff'))

    eeg_data = pd.DataFrame(data)

    eeg_data['eyeDetection'] = eeg_data['eyeDetection'].str.decode("utf-8")

    labels = eeg_data.pop('eyeDetection')

    train_labels = pd.get_dummies(labels)

    train_labels = train_labels.values

    x_train, x_test, y_train, y_test = train_test_split(eeg_data, train_labels, test_size = test_size)

    return x_train, x_test, y_train, y_test

