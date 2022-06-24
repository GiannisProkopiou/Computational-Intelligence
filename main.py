from DeepHelpers import get_emdeding_data, make_clean_dataset, Read_Two_Column_File, evaluate_model, get_scaled_data, new_vocab
from DeepModels import get_embedding_model, get_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def main():

    #A.1

    # read the vocabulary and the indexes
    vocabulary, indexes = Read_Two_Column_File('Data/vocabs.txt')

    # new vocab
    most_important_words = np.load("best_sol.npy", allow_pickle=True)
    vocabulary = new_vocab(most_important_words)

    # make a clean dataset
    # make_clean_dataset('Data/train-data.dat', 'Data/train-data-cleaned.dat')
    # make_clean_dataset('Data/test-data.dat', 'Data/test-data-cleaned.dat')

    # get the scaled data
    X_train, maxlen_train = get_scaled_data('Data/train-data-cleaned.dat', vocabulary, scale='standard')
    X_test, max_len_test = get_scaled_data('Data/test-data-cleaned.dat', vocabulary, scale='standard')

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # X_train_embedding, maxlen_train_embedding = get_emdeding_data('Data/train-data-cleaned.dat')
    # X_test_embedding, max_len_test_embedding = get_emdeding_data('Data/test-data-cleaned.dat')
    #
    # X_train_embedding = np.array(pad_sequences(X_train_embedding, padding='post', maxlen=maxlen_train_embedding))
    # X_test_embedding = np.array(pad_sequences(X_test_embedding, padding='post', maxlen=maxlen_train_embedding))


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


    # d)
    lr = 1e-3
    m = 0.6
    l2 = 0.5
    n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
    model = get_model(n_inputs, n_outputs, lr, m, l2)
    evaluate_model(X_train, y_train, X_test, y_test, model)

    #A.5 - Bonus
    # n_inputs, n_outputs = X_train_embedding.shape[1], y_train.shape[1]
    # model = get_embedding_model(maxlen_train_embedding, n_inputs, n_outputs, 0.001, 0.6, 0.9)
    # evaluate_model(X_train_embedding, y_train, X_test_embedding, y_test, model)


if __name__ == '__main__':
    main()

