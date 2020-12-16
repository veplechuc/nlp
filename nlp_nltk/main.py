from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

text = "Computers don't speak English. So, we've to learn C, C++, ,C#, Java, Python and the like! Yay!. Or may we should use another tool"

#tokenize sentence
sentences = sent_tokenize(text)
#print(len(sentences), 'sentences:', sentences)

#word tokenize
words = word_tokenize(text)
print(len(words), 'words:', words)

#stop words
stop_words = stopwords.words('english')
#print(len(stop_words), "stopwords:", stop_words)

words = [word for word in words if word not in stop_words]
print(len(words), "without stopwords:", words)

punctuations = list(string.punctuation)

#print(punctuations)
words = [word for word in words if word not in punctuations]
print(len(words), "words without stopwords and punctuations:", words)