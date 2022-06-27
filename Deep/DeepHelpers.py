import re
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


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
        if int(word) < len(vocabulary):
            bow_word_counts[int(word)] += 1

    # return the word_counts dictionary
    return bow_word_counts, len(document.split(" ")[1:])


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

    vector = np.array(vector)
    if scale == 'standard':
        scaler = preprocessing.StandardScaler().fit(vector)
        vector = scaler.transform(vector)
    elif scale == 'normalize':
        scaler = preprocessing.Normalizer().fit(vector)
        vector = scaler.transform(vector)

    return np.squeeze(vector)


def get_scaled_data(dataset, vocabulary, scale='standard'):

    bow_word_counts = []

    maxlen = 0

    with open(f'{dataset}', 'r') as data_cleaned:
        for line in data_cleaned:
            bow_word_count, length = bag_of_words(vocabulary, line)
            bow_word_counts.append(bow_word_count)

            if length > maxlen:
                maxlen = length

    X = vector_scaling(bow_word_counts, scale)

    return X, maxlen


def get_emdeding_data(dataset, vocabulary):

    X = []
    maxlen = 0

    with open(f'{dataset}', 'r') as data_cleaned:
        for line in data_cleaned:
            line = line.split('\n')[0]
            words = [int(word) for word in line.split(' ')[1:] if word in vocabulary]
            X.append(words)

            length = len(line.split(' ')[1:])
            if length > maxlen:
                maxlen = length

    return X, maxlen


# function to create new vocab from the output of the genetic algorithm
def new_vocab(most_important_words):

    indices = [i for i, x in enumerate(most_important_words.tolist()) if x == 1]
    return indices


def get_y(train_labels_file='Data/train-label.dat', test_bales_file='Data/test-label.dat'):
    # get the labels
    y_train = []
    with open(train_labels_file, 'r') as train_labels:
        for line in train_labels:
            line = line.split("\n")[0]
            elems = [int(elem) for elem in line.split(' ')]
            y_train.append(np.array(elems))

    y_test = []
    with open(test_bales_file, 'r') as test_labels:
        for line in test_labels:
            line = line.split("\n")[0]
            elems = [int(elem) for elem in line.split(' ')]
            y_test.append(np.array(elems))

    y_test = np.array(y_test)
    y_train = np.array(y_train)

    return y_test, y_train


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, X_test, y_test, model):

    results = list()
    ces = []
    mses = []
    accs = []
    # define evaluation procedure
    cv = KFold(n_splits=5)
    # enumerate folds
    for fold, (train_ix, test_ix) in enumerate(cv.split(X)):
        # prepare data
        X_train, X_test_fold = X[train_ix], X[test_ix]
        y_train, y_test_fold = y[train_ix], y[test_ix]
        # define model
        # fit model
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-05, verbose=0)
        early_stoping = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto",
                                      baseline=None,
                                      restore_best_weights=True)
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        history = model.fit(X_train, y_train, validation_data=(X_test_fold, y_test_fold), batch_size=128, epochs=100,
                            verbose=1, callbacks=[early_stoping])

        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        fig, axs = plt.subplots(5, figsize=(20, 20))


        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_title('model accuracy')
        axs[0].set_ylabel('accuracy')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'test'], loc='upper left')
        # axs[0].show()

        axs[1].plot(history.history['categorical_crossentropy'])
        axs[1].plot(history.history['val_categorical_crossentropy'])
        axs[1].set_title('model categorical_crossentropy')
        axs[1].set_ylabel('categorical_crossentropy')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'test'], loc='upper left')
        # axs[1].show()

        axs[2].plot(history.history['mean_squared_error'])
        axs[2].plot(history.history['val_mean_squared_error'])
        axs[2].set_title('model mean_squared_error')
        axs[2].set_ylabel('mean_squared_error')
        axs[2].set_xlabel('epoch')
        axs[2].legend(['train', 'test'], loc='upper left')
        # axs[2].show()

        axs[3].plot(history.history['categorical_accuracy'])
        axs[3].plot(history.history['val_categorical_accuracy'])
        axs[3].set_title('model categorical_accuracy')
        axs[3].set_ylabel('categorical_accuracy')
        axs[3].set_xlabel('epoch')
        axs[3].legend(['train', 'test'], loc='upper left')
        # axs[3].show()

        axs[4].plot(history.history['loss'])
        axs[4].plot(history.history['val_loss'])
        axs[4].set_title('model loss')
        axs[4].set_ylabel('loss')
        axs[4].set_xlabel('epoch')
        axs[4].legend(['train', 'test'], loc='upper left')
        # axs[4].show()

        fig.savefig(f'{fold}_fold_plot.png')

        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)

        ces.append(history.history['categorical_crossentropy'][-1])
        mses.append(history.history['mean_squared_error'][-1])
        accs.append(history.history['categorical_accuracy'][-1])


    print(ces)
    print(mses)
    print(accs)

    return results