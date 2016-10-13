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

print(len(poo))
print(len(lan))
print(len(yoo))

with open('yoyo.csv', mode="w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    headerz = ['1', '2', '3']
    spamwriter.writerow(headerz)
    x = 1
    while x < len(poo):
        rowz = [poo[x], lan[x], yoo[x]]
        spamwriter.writerow(rowz)
        x += 1

nltk.help.upenn_tagset('IN')
nltk.help.upenn_tagset('NNP')
nltk.help.upenn_tagset('RB')

print(porter().stem("ponies"))
print(wnl.lemmatize("ponies"))

print(porter().stem("equivalent"))
print(wnl.lemmatize("equivalent"))

print(porter().stem("example"))
print(wnl.lemmatize("example"))
