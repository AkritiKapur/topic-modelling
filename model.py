
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation

beta = 0.01
alpha = 0.1

ALPHABET_SIZE = 26

TOPIC_DIST = [alpha] * 3
WORD_DIST = [beta] * 20
WORDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


def get_training_data(N, tokens, words, topics, beta):
    """
        Generative model to get synthetic data
    :param N: No of documents
    :param tokens: No of tokens in a single document
    :param words: Vocabulary size
    :param topics: No of topics
    :param beta: Beta distribution over words given topic
    :return: synthetic data, word-topic distribution, topic-doc distribution
    """

    # Dirichlet distribution over three topics/ document
    t_doc_samples = np.random.dirichlet(TOPIC_DIST, N)
    word_t_samples = np.random.dirichlet(WORD_DIST, topics)

    print(word_t_samples)
    print()
    training_data = []

    for sample in t_doc_samples:
        doc = ''

        # Generate tokens
        for _ in range(tokens):

            # Choose a topic from the multinomial distribution
            topic_dis = np.random.multinomial(1, sample, size=1)
            topic_index = np.argmax(topic_dis[0])

            # Choose a word from the topic(selected) multinomial distribution
            word_dist = np.random.multinomial(1, word_t_samples[topic_index], size=1)
            word_index = np.argmax(word_dist[0])
            letter = chr(word_index + 65)

            # Add letter to the doc
            doc += letter + ' '

        # Add doc to the synthetic data
        training_data.append(doc[:-1])

    print(training_data)
    return training_data, word_t_samples, t_doc_samples


def get_feature_vector_for_letter_data(data):
    feature_list = [0 for i in range(20)]
    letters = data.split()
    for letter in letters:
        feature_list[ord(letter) - 65] += 1

    return feature_list


def part_2(X, alpha, beta):
    LDA_model = LatentDirichletAllocation(n_components=3, doc_topic_prior=alpha, topic_word_prior=beta)

    train_data = []
    for data in X:
        feature_vector = get_feature_vector_for_letter_data(data)
        train_data.append(feature_vector)

    LDA_model.fit(train_data)
    doc_topic_dist = LDA_model.transform(train_data)
    # print_top_words(LDA_model)
    prob = LDA_model.components_ / LDA_model.components_.sum(axis=1)[:, np.newaxis]

    return prob, doc_topic_dist


def print_top_words(model):

    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d: " % topic_idx)
        print(" ".join([chr(i+65)
                        for i in topic.argsort()[:-20 - 1:-1]]))
    print()


def compare_dist(actual_dist, topic_model_dist):
    x = np.arange(20)
    plt.title("distribution accross words topics")

    for i in range(3):
        word_dist_act = actual_dist[i]
        word_dist_lda = topic_model_dist[i]

        plt.subplot(2, 1, 1)
        act_line, = plt.plot(x, word_dist_act, color='green')
        plt.xticks(np.arange(20), WORDS)
        plt.title("True distribution")
        plt.ylabel("P(W|T)")
        plt.xlabel("word")

        plt.subplot(2, 1, 2)
        lda_line, = plt.plot(x, word_dist_lda, color='red')
        plt.xticks(np.arange(20), WORDS)
        plt.title("Recovered distribution")
        plt.ylabel("P(W|T)")
        plt.xlabel("word")


    plt.show()


def cal_entropy(syn_data):
    alpha_list = [1, 5, 10, 20, 30, 50, 100]

    entropy_list = []
    for alpha in alpha_list:
        _, dist = part_2(syn_data, alpha, beta)
        dist = np.array(dist)
        entropy = np.sum(dist * np.log(dist))
        entropy_list.append(-1 * entropy / 200)

    plt.xticks(np.arange(len(alpha_list)), alpha_list)
    plt.xlabel("alpha")
    plt.ylabel("entropy")
    plt.title("variation of entropy with change in alpha")
    plt.plot(np.arange(len(alpha_list)), entropy_list)
    plt.show()


def cal_word_entropy(syn_data):
    beta_list = [0.01, 0.1, 1, 10, 100, 1000]
    entropy_list = []

    for beta in beta_list:
        dist, _ = part_2(syn_data, alpha, beta)
        entropy = 0
        for topic in range(len(dist)):
            for word in range(len(dist[topic])):
                entropy += dist[topic][word] * (np.log(dist[topic][word]) if dist[topic][word] else 1)

        entropy = -1 * entropy / len(dist)
        entropy_list.append(entropy)

    plt.xticks(np.arange(len(beta_list)), beta_list)
    plt.xlabel("beta")
    plt.ylabel("entropy")
    plt.title("variation of entropy with change in beta")
    plt.plot(np.arange(len(beta_list)), entropy_list)
    plt.show()


if __name__ == '__main__':
    synthetic_data, dist, topic_dist = get_training_data(200, 50, 20, 3, 0.01)
    lda_dist, _ = part_2(synthetic_data, alpha, beta)
    compare_dist(dist, lda_dist)

    cal_word_entropy(synthetic_data)
    cal_entropy(synthetic_data)
