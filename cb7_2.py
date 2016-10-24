import collections
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def bag_of_words(words):
    return dict([(word, True) for word in words])


def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    words_list = [word for word in words]
    return bag_of_non_stopwords(words_list + bigrams)


def label_feats_from_corpus(corp, feature_detector=bag_of_bigrams_words):
    label_feats = collections.defaultdict(list)
    # For Every Category in the Corpus
    for label in corp.categories():

        # For every file in all of the files with the category label
        for fileid in corp.fileids(categories=[label]):

            # add feutures from files to bagofwords
            feats = feature_detector(corp.words(fileids=[fileid]))

            # Add features to label_feats
            label_feats[label].append(feats)
    return label_feats


# function that splits a list dicts at a cutoff percentage.
def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        print(feats)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


lfeats = label_feats_from_corpus(movie_reviews, bag_of_words)

train_feats, test_feats = split_label_feats(lfeats, split=0.75)

nb_classifier = NaiveBayesClassifier.train(train_feats)
print(accuracy(nb_classifier, test_feats))
nb_classifier.show_most_informative_features()
