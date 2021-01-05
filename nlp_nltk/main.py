from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize,PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
import string

text = "Computers don't speak or speaking in English. So, we've to learn, or learning C, C++, ,C#, Java, Python and the like! Yay!. Or may we should use another tool"

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

#stemming stemm the root of the word
ps = PorterStemmer()

stemmed_example = [ps.stem(word) for word in words ] # remember to tokenize the text first

print (f'Stemming -> {stemmed_example}')

# tagging
# need to have installed
# nltk.download('averaged_perceptron_tagger')
# nltk.download('state_union')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = word_tokenize(i)
            tagged = pos_tag(words)
            #using chunck
            # print(tagged)
            # chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            # chunkParser = RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            # print(chunked)
            # for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
            #     print(subtree)
            # chunked.draw()     
            namedEnt = ne_chunk(tagged, binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()

############COMMENTS
# NE Type and Examples
# ORGANIZATION - Georgia-Pacific Corp., WHO
# PERSON - Eddy Bonte, President Obama
# LOCATION - Murray River, Mount Everest
# DATE - June, 2008-06-29
# TIME - two fifty a m, 1:30 p.m.
# MONEY - 175 million Canadian Dollars, GBP 10.40
# PERCENT - twenty pct, 18.75 %
# FACILITY - Washington Monument, Stonehenge
# GPE - South East Asia, Midlothian