from loguru import logger
import pickle
from nltk.corpus import movie_reviews
from nltk import FreqDist
import random

def save_model(model, file_name):
    logger.info("Saving model...")
    with open(f"{file_name}.pickle", "wb") as save_classifier:
        pickle.dump(model, save_classifier)


def load_model(file_name):
    logger.info("Loading model...")
    with open(f"{file_name}.pickle", "rb") as classifier_f:
        classifier = pickle.load(classifier_f)
    return classifier

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
    all_words = FreqDist(all_words)
    # print(all_words.most_common(15))
    # word = 'bad'
    # print(f'how many times {word} appears on the text --> {all_words[word]}')
    return all_words
