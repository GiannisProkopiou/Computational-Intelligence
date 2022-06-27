from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random


# function to clean initial dataset and make it like a bag of words
def make_clean_dataset(dataset, new_dataset_name):

    # clean dataset
    with open(f'{dataset}', 'r') as data:
        pattern = re.compile("<(.*?)>")
        lines = []
        for line in data:
            line = pattern.sub("", line)
            line = re.sub(' +', ' ', line)
            lines.append(line)

    with open(f'{new_dataset_name}', 'w') as data_cleaned:
        data_cleaned.writelines(lines)


# function to read the vocabulary
def Read_Two_Column_File(file_name):

    with open(file_name, 'r') as data:
        vocabulary = []
        for line in data:
            p = line.split()
            vocabulary.append(str(p[0]).split(',')[0])

    return vocabulary


# returns an array with o and 1 of length atom_length
def generate_atom(atom_length):
    return random.choices([0, 1], k=atom_length)


# generates a population of atoms given a number of atoms in population
def generate_population(population_length, atom_bits):

    population = []
    for _ in range(population_length):
        population.append(generate_atom(atom_bits))

    return population


# function to get corpus from coded dataset
def get_corpus(corpus_file, vocabulary):

    corpus = []

    with open(corpus_file, 'r') as cor_file:
        for line in cor_file:
            coded_words = [int(word) for word in line.split()]
            decoded_words = [vocabulary[word] for word in coded_words]
            sub_corpus = " ".join(decoded_words)
            corpus.append(sub_corpus)

    return corpus


# function that calculates the tfidf score for each corpus and word
def tfidf_score(corpus):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    return vectorizer.get_feature_names_out(), X.toarray()


# make a dict with each word and it's score
def make_tfidf_score_dict(vocabulary, vectorizer_feature_names, X_toarray, corpus_len):

    tfidf_score_dict = {}

    for word in vocabulary:
        score = 0
        for corpus_doc in range(corpus_len):
            temp_dict = dict(zip(vectorizer_feature_names, X_toarray[corpus_doc]))
            if word in temp_dict.keys():
                score += temp_dict[word]

        tfidf_score_dict[word] = score
        print(f'Word: {word}')
        print(f'Score: {score}')

    return tfidf_score_dict


# fitness function to serve as the objective of the genetic algorithm
def fitness(atom,  vocabulary, tfidf_score_dict, corpus_len):

    word_counter = sum(atom)

    # discard non ligal solutions
    if word_counter < 1000:
        return 0

    fitness_score = 0
    for at in range(len(atom)):
        if atom[at] == 1:
            word_in_atom = vocabulary[at]
            # average tfidf score
            fitness_score += tfidf_score_dict[word_in_atom] / corpus_len

    # penalty with respect to the score and number of words
    if word_counter > 1000:
        penalty = ((word_counter - 1000) / 7520) * fitness_score

    return fitness_score - penalty


