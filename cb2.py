from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import re
import enchant
from nltk.metrics import edit_distance
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




class SpellingReplacer(object):
    def __init__(self, dict_name='en', max_dist=4):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

replacer = RepeatReplacer()
replacer2 = SpellingReplacer()
print(replacer.replace('goooose'))
print(replacer2.replace('lunguag'))

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

replacer3 = WordReplacer({'bday': 'birthday'})
print(replacer3.replace('bday'))
