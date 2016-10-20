from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

syn = wordnet.synsets('cookbook')[0]
syn_name = syn.name()
syn_def = syn.definition()

print('synname: %s, syndef: %s' % (syn_name, syn_def))

print(syn.hypernym_paths())

lemmas = syn.lemmas()
lemmas_length = len(lemmas)
lemmas[0].name
tokenizer = RegexpTokenizer("[\w']+")
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))

filter_stops = lambda w: len(w) < 3 or w in stopset

tokenz = tokenizer.tokenize(webtext.raw('grail.txt'))

words = [w.lower() for w in tokenz]
bcf = BigramCollocationFinder.from_words(words)
bcf.apply_word_filter(filter_stops)
bigrams = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
print(bigrams)