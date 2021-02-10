from loguru import logger
import pickle

def save_model(model, file_name):
    logger.info("Saving model...")
    with open(f"{file_name}.pickle", "wb") as save_classifier:
        pickle.dump(model, save_classifier)


def load_model(file_name):
    logger.info("Loading model...")
    with open(f"{file_name}.pickle", "rb") as classifier_f:
        classifier = pickle.load(classifier_f)
    return classifier
