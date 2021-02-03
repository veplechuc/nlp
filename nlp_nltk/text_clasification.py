from loguru import logger
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        return mode(votes)

    def confidence(self, features):
        votes = []
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

"remember --> nltk.download('movie_reviews')"

"""In each category (we have pos or neg), take all of the file IDs (each review has its own ID),
then store the word_tokenized version (a list of words) for the file ID, followed by the positive or negative label in one big list
"""
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

" find these top 3,000 words in our positive and negative documents, marking their presence as either positive or negative:"
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#save model
def save_model(model, file_name):
    logger.info("Saving model...")
    with open(f'{file_name}.pickle', "wb") as save_classifier:
        pickle.dump(model, save_classifier)
    
#load the model saved
def load_model(file_name):
    logger.info("Loading model...")
    with open(f'{file_name}.pickle', "rb") as classifier_f:
        classifier = pickle.load(classifier_f)
    return classifier
# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# word = 'bad'
# print(f'how many times {word} appears on the text --> {all_words[word]}')

logger.info("Create word features...")
word_features = list(all_words.keys())[:3000]

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# uncomment this line if you dont have the model saved
#classifier = nltk.NaiveBayesClassifier.train(training_set)

# now call the saved method comment if this is the first time you run this program
NB_classifier = load_model('naive_bayes2')


print("Classifier NaiveBayes accuracy percent:",(nltk.classify.accuracy(NB_classifier, testing_set))*100)
# what the most valuable words are when it comes to positive or negative reviews:
# classifier.show_most_informative_features(15)
# save_model(classifier, 'naive_bayes2')

models = [MultinomialNB, BernoulliNB, LogisticRegression,
                       SGDClassifier, SVC, LinearSVC, NuSVC
                     ]

def classify(models):
    for model in models:
        classifier = SklearnClassifier(model())
        classifiers.append(classifier)
        classifier.train(training_set)
        print(f'{model.__name__}_classifier accuracy percent:', (nltk.classify.accuracy(classifier, testing_set))*100)

#adding the first classifier
classifiers = [NB_classifier]

logger.info("print accuracy...")
classify(models)

voted_classifier = VoteClassifier(*classifiers)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
for x in range(0,5):
    print("Classification:", voted_classifier.classify(testing_set[x][0]), 
            "Confidence %:",voted_classifier.confidence(testing_set[x][0])*100)
