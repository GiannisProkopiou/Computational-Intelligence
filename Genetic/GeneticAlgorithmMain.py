# imports
import numpy as np
from GeneticHelpers import Read_Two_Column_File, generate_population, get_corpus, fitness
from GeneticOperators import selection, crossover, mutation
import matplotlib.pyplot as plt


# genetic algorithm
def genetic_algorithm(objective, vocabulary, tfidf_score_dict, corpus_len, pop, n_iter, r_cross, r_mut, patience):

    # array to store performance scores
    performance = []

    # keep track of best solution
    best, best_eval = 0, objective(pop[0], vocabulary, tfidf_score_dict, corpus_len)

    # enumerate generations
    gens_with_no_better_score = 0  # patience counter
    for gen in range(n_iter):

        # evaluate all candidates in the population
        scores = [objective(c, vocabulary, tfidf_score_dict, corpus_len) for c in pop]

        # check for new best solution
        gens_with_no_better_score += 1
        for i in range(len(pop)):
            if scores[i] > best_eval:
                # difference between bests to see improvement
                diff_percent = (scores[i] / best_eval) - 1
                if diff_percent < 0.1:
                    # if really better re-initialize patience
                    gens_with_no_better_score = 0

                best, best_eval = pop[i], scores[i]
                print(f"Gen: {gen} - New best atom: {i} - Score: {scores[i]}")

        # if patience reached, stop
        if gens_with_no_better_score == patience:
            break

        # elitism - keeping the best each time
        max_score_idx = scores.index(max(scores))
        max_score = scores.pop(max_score_idx)
        atom_with_max_score = pop.pop(max_score_idx)

        second_max_score_idx = scores.index(max(scores))
        second_max_score = scores.pop(second_max_score_idx)
        atom_with_second_max_score = pop.pop(second_max_score_idx)

        # select parents
        selected = [selection(pop, scores) for _ in range(len(pop))]

        # create the next generation
        children = list()
        children.append(atom_with_max_score)  # new gen with the best without changes
        children.append(atom_with_second_max_score)  # new gen with the second best without changes

        for i in range(0, len(pop), 2):

            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]

            # crossover and mutation (with elitism)
            new_children = crossover(p1, p2, r_cross)

            for c in new_children:
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)

        # replace population
        pop = children

        performance.append(best_eval)

    print(f"Best: {best} - Best Eval: {best_eval} - Number of Words: {sum(best)}")
    return best, best_eval, performance, pop


# Dataset clean and read
# make_clean_dataset('Data/train-data.dat', 'Data/train-data-cleaned.dat')
vocabulary = Read_Two_Column_File('Data/vocabs.txt')

# Population Genaration
population_len = 300

# length of atom
atom_bits = 8520

# population generation
population = generate_population(population_len, atom_bits)

# Corpus generation and tfidf scoring
corpus = get_corpus("Data/train-data-cleaned.dat", vocabulary)

# tfidf calculation
# vectorizer_feature_names, X_toarray = tfidf_score(corpus)

# dictionaries to not recalculate every time
# tfidf_score_dict = make_tfidf_score_dict(vocabulary, vectorizer_feature_names, X_toarray, len(corpus))

# dave the dictionary to use
# np.save('tfidf_score_dict.npy', tfidf_score_dict, allow_pickle=True)
tfidf_score_dict = np.load('Data/tfidf_score_dict.npy', allow_pickle=True).item()

# define the total iterations
n_iter = 1000

# crossover rate
r_cross = 0.7

# mutation rate
r_mut = 1.0 / float(atom_bits)
# r_mut = 0.10

# corpus len
corpus_len = len(corpus)

# patience in gens with the same score
patience = 10

# use of the genetic algorithm
best, best_eval, performance, best_pop = genetic_algorithm(fitness, vocabulary, tfidf_score_dict, corpus_len,
                                                           population, n_iter, r_cross, r_mut, patience)

# dataset creation based on the best population of the genetic algorithm
genetic_dataset = np.array(best_pop)
# save the best population to use
# np.save('../Data/genetic_dataset.npy', genetic_dataset, allow_pickle=True)

# plot the performance/gens each time
plt.title("Performance / Gens")
plt.xlabel("Gens")
plt.ylabel("Performance")
plt.plot(performance, color="red")
plt.show()





