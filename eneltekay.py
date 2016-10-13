import nltk
import csv
from nltk import word_tokenize as tokenoise
from nltk.stem.porter import PorterStemmer as porter

sentence = "at eight o'clock on THursday morning Arthur didn't feel very good."
raw = "DENNIS: Listen, strange women lying in ponds disributing swords is no basis for a system of government. Supreme executive power derives from a mandate from the masses, not from some farcial aquatic cerenomy."

wnl = nltk.WordNetLemmatizer()
lancaster = nltk.LancasterStemmer()

tokens = tokenoise(sentence)
tokens
tagged = nltk.pos_tag(tokens)

tokens2 = tokenoise(raw)
tokens2


poo = [porter().stem(t) for t in tokens2]
lan = [lancaster.stem(t) for t in tokens2]
yoo = [wnl.lemmatize(t) for t in tokens2]

with open('yoyo.csv', mode="wt") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t',
        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writeheader('1', '2', '3')
    for p in poo:
        spamwriter.writerow(p)

nltk.help.upenn_tagset('IN')
nltk.help.upenn_tagset('NNP')
nltk.help.upenn_tagset('RB')

print(porter().stem("ponies"))
print(wnl.lemmatize("ponies"))

print(porter().stem("equivalent"))
print(wnl.lemmatize("equivalent"))

print(porter().stem("example"))
print(wnl.lemmatize("example"))
