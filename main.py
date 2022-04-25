import re
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# make a function that implements the bag of words model with a given vocabulary
# and a given document
# a)
def bag_of_words(vocabulary, document):
    # initialize a dictionary to store the number of times each word appears
    # in the document
    bow_word_counts = [0 for i in range(len(vocabulary))]
    # iterate over the words in the document
    for word in document.split(" ")[1:]:
        # if the word is in the vocabulary, increment its count in the
        # word_counts dictionary
        bow_word_counts[int(word)] += 1

    # return the word_counts dictionary
    return bow_word_counts

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        vocabulary = []
        indexes = []
        for line in data:
            p = line.split()
            vocabulary.append(str(p[0]).split(',')[0])
            indexes.append(int(p[1]))

    return vocabulary, indexes

def make_clean_dataset(dataset, new_dataset_name):
    with open(f'{dataset}', 'r') as data:
        pattern = re.compile("<(.*?)>")
        lines = []
        for line in data:
            line = pattern.sub("", line)
            line = re.sub(' +', ' ', line)
            lines.append(line)

    with open(f'{new_dataset_name}', 'w') as data_cleaned:
        data_cleaned.writelines(lines)

# b)
def vector_scaling(vector, scale='standard'):
    vector = np.array(vector).reshape(-1, 1)
    if scale == 'standard':
        scaler = preprocessing.StandardScaler().fit(vector)
        vector = scaler.transform(vector)
    elif scale == 'normalize':
        scaler = preprocessing.Normalizer().fit(vector)
        vector = scaler.transform(vector)

    return np.squeeze(vector)

def get_scaled_data(dataset, indexes, scale='standard'):

    X = []

    with open(f'{dataset}', 'r') as data_cleaned:
        for line in data_cleaned:
            bow_word_counts = bag_of_words(indexes, line)
            train_scaled_bow_vector = vector_scaling(bow_word_counts, 'standard')
            X.append(train_scaled_bow_vector)

    return X

# get the model
def get_model(n_inputs, n_outputs):
    model = keras.Sequential()
    model.add(layers.Dense(20, input_dim=n_inputs, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)

    print(model.summary())

    return model

# evaluate a model using repeated k-fold cross-validation
# def evaluate_model(X, y):
#     results = list()
#     n_inputs, n_outputs = X.shape[1], y.shape[1]
#     # define evaluation procedure
#     cv = KFold(n_splits=5)
#     # enumerate folds
#     for train_ix, test_ix in cv.split(X):
#         # prepare data
#         X_train, X_test = X[train_ix], X[test_ix]
#         y_train, y_test = y[train_ix], y[test_ix]
#         # define model
#         model = get_model(n_inputs, n_outputs)
#         # fit model
#         model.fit(X_train, y_train, verbose=0, epochs=100)
#         # make a prediction on the test set
#         yhat = model.predict(X_test)
#         # round probabilities to class labels
#         yhat = yhat.round()
#         # calculate accuracy
#         acc = accuracy_score(y_test, yhat)
#         # store result
#         print('>%.3f' % acc)
#         results.append(acc)
#     return results


def main():

    #A.1

    # read the vocabulary and the indexes
    vocabulary, indexes = Read_Two_Column_File('Data/vocabs.txt')

    # make a clean dataset
    # make_clean_dataset('Data/train-data.dat', 'Data/train-data-cleaned.dat')
    # make_clean_dataset('Data/test-data.dat', 'Data/test-data-cleaned.dat')

    # get the scaled data
    X_train = get_scaled_data('Data/train-data-cleaned.dat', indexes, scale='standard')
    X_test = get_scaled_data('Data/test-data-cleaned.dat', indexes, scale='standard')

    # get the labels
    y_train = []
    with open('Data/train-label.dat', 'r') as train_labels:
        for line in train_labels:
            line = line.split("\n")[0]
            elems = [int(elem) for elem in line.split(' ')]
            y_train.append(np.array(elems))

    y_test = []
    with open('Data/test-label.dat', 'r') as test_labels:
        for line in test_labels:
            line = line.split("\n")[0]
            elems = [int(elem) for elem in line.split(' ')]
            y_test.append(np.array(elems))

    y_test = np.array(y_test)
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # c) splits the data into 5 folds
    cv = KFold(n_splits=5)
    train_splits = cv.split(X_train)
    test_splits = cv.split(X_test)
    y_train_splits = cv.split(y_train)
    y_test_splits = cv.split(y_test)

    # d)
    model = get_model(np.shape(X_train)[1], np.shape(y_train)[1])
    model.fit(X_train, y_train, verbose=0, epochs=100)
    plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
    history = model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score)
    # print("Test Accuracy:", score[1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()

