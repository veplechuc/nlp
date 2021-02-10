from vote_classifier import VoteClassifier
from utils import save_model, load_model
from loguru import logger
import nltk
import random
from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
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


def get_documents():
    """In each category (we have pos or neg), take all of the file IDs (each review has its own ID),
    then store the word_tokenized version (a list of words) for the file ID, followed by the positive or negative label in one big list
    """
    documents = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    return random.shuffle(documents)


def all_words_freq():
    all_words = [w.lower() for w in movie_reviews.words()]
    all_words = nltk.FreqDist(all_words)
    # print(all_words.most_common(15))
    # word = 'bad'
    # print(f'how many times {word} appears on the text --> {all_words[word]}')
    return all_words


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


# print(documents[1])


def feature_set():
    featuresets = [
        (find_features(rev), category) for (rev, category) in get_documents()
    ]
    return featuresets


# set that we'll train our classifier with
training_set = feature_set()[:ELEMENTS]

# set that we'll test against.
testing_set = feature_set()[ELEMENTS:]

# uncomment this line if you dont have the model saved
# classifier = nltk.NaiveBayesClassifier.train(training_set)

# now call the saved method --- comment if this is the first time you run this program
NB_classifier = load_model("naive_bayes2")


print(
    "Classifier NaiveBayes accuracy percent:",
    (nltk.classify.accuracy(NB_classifier, testing_set)) * 100,
)
# what the most valuable words are when it comes to positive or negative reviews:
# classifier.show_most_informative_features(15)
# save_model(classifier, 'naive_bayes2')

# @njit(parallel=True)
def classify(models):
    for model in models:
        classifier = SklearnClassifier(model())
        classifiers.append(classifier)  # adding the rest of classifiers
        classifier.train(training_set)
        print(
            f"{model.__name__}_classifier accuracy percent:",
            (nltk.classify.accuracy(classifier, testing_set)) * 100,
        )


# adding the first classifier
classifiers = [NB_classifier]

logger.info("print accuracy...")
classify(models)


def list_confidence():
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
