from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet

stemmer = PorterStemmer()
print(stemmer.stem('cooking'))
print(stemmer.stem('cookery'))
stemmer2 = LancasterStemmer()

print(stemmer2.stem('cooking'))
print(stemmer2.stem('cookery'))

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('cooking'))

class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


replacer = RepeatReplacer()

print(replacer.replace('goooose'))

