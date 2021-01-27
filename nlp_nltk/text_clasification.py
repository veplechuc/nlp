import nltk
import random
from nltk.corpus import movie_reviews

"remember --> nltk.download('movie_reviews')"

"""In each category (we have pos or neg), take all of the file IDs (each review has its own ID),
then store the word_tokenized version (a list of words) for the file ID, followed by the positive or negative label in one big list
"""
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
word = 'bad'
print(f'how many times {word} appears on the text --> {all_words[word]}')