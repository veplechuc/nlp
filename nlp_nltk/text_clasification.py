from vote_classifier import VoteClassifier
from utils import save_model, load_model, get_documents, all_words_freq
from loguru import logger

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from statistics import mode

"remember --> nltk.download('movie_reviews')"


models = [
    MultinomialNB,
    BernoulliNB,
    LogisticRegression,
    SGDClassifier,
    SVC,
    LinearSVC,
    NuSVC,
]

TOP_WORDS = 3000
ELEMENTS = 1900

def word_features():
    logger.info("Create word features from movie_reviews...")
    all_words = all_words_freq()
    word_features = list(all_words.keys())[:TOP_WORDS]
    # print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
    return word_features


def find_features(document):
    """find these top 3,000 words in our positive and negative documents,
    marking their presence as either positive or negative:
    """
    logger.info("features in word_features list...")
    words = set(document)
    features = {}
    for w in word_features():
        features[w] = w in words

    return features


def feature_set():
    featuresets = [
        (find_features(rev), category) for (rev, category) in get_documents()
    ]
    return featuresets


def list_confidence(classifiers, training_set, testing_set):
    voted_classifier = VoteClassifier(*classifiers)
    logger.info(
        "voted_classifier accuracy percent: ",
        (nltk.classify.accuracy(voted_classifier, testing_set)) * 100,
    )
    # print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
    for x in range(0, 10):
        print(
            "Classification:",
            voted_classifier.classify(testing_set[x][0]),
            "Confidence %:",
            voted_classifier.confidence(testing_set[x][0]) * 100,
        )

# @njit(parallel=True)
def classify(models, classifiers, training_set, testing_set):
    for model in models:
        classifier = SklearnClassifier(model())
        classifiers.append(classifier)  # adding the rest of classifiers
        classifier.train(training_set)
        print(
            f"{model.__name__}_classifier accuracy percent:",
            (nltk.classify.accuracy(classifier, testing_set)) * 100,
        )

def naive_bayes_classifier(training_set):
    # uncomment this line if you dont have the model saved
    return nltk.NaiveBayesClassifier.train(training_set)


def proccess():

    # set that we'll train our classifier with
    training_set = feature_set()[:ELEMENTS]

    # set that we'll test against.
    testing_set = feature_set()[ELEMENTS:]

    if not exists_file:
        # uncomment this line if you dont have the model saved
        NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
        save_model(NB_classifier, 'naive_bayes2')

    # now call the saved method 
    NB_classifier = load_model("naive_bayes2")

    print(
        "Classifier NaiveBayes accuracy percent:",
        (nltk.classify.accuracy(NB_classifier, testing_set)) * 100,
    )
    # what the most valuable words are when it comes to positive or negative reviews:
    # classifier.show_most_informative_features(15)


    # adding the first classifier
    classifiers = [NB_classifier]
    logger.info("print accuracy...")
    classify(models, classifiers, training_set, testing_set)

    list_confidence(classifiers, training_set, testing_set)


if __name__ == "__main__":
    proccess()