from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def bag_of_words(words):
    return dict([(word, True) for word in words])


def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)


def bag_of_bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_non_stopwords(words + bigrams)


wordz = ['the', 'quick', 'brown', 'fox']
baggie = bag_of_words(wordz)
print(baggie)

baggo = bag_of_words_not_in_set(wordz, ['the'])

print(baggo)

baggy = bag_of_non_stopwords(wordz)
print(baggy)


print(bag_of_bigram_words(wordz))

print()
