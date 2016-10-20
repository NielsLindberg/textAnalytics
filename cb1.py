import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.corpus import webtext

para = "Hello World. It's good to see you. Thanks for buying this book"


print(sent_tokenize(para))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print(tokenizer.tokenize(para))

print(word_tokenize('Hello World.'))

tokenizer = RegexpTokenizer("[\w']+")
print(tokenizer.tokenize("Can't is a contraction."))

text = webtext.raw('overheard.txt')
sent_tokenizer = PunktSentenceTokenizer(text)
sents1 = sent_tokenizer.tokenize(text)
print(sents1[0])

sents2 = sent_tokenizer.tokenize(text)
print(sents2[0])

print(sents1[678])
print(sents2[678])

english_stops = set(stopwords.words('english'))
words = ["Can't", 'is', 'a', 'contraction']
print([word for word in words if word not in english_stops])