from nltk.corpus import wordnet as wn
import nltk
import sys


def synsetSimilarity(s1,s2):
    synset1 = [perfectSynset(word,synset1)]
    return synsetSimilarityRatio


def perfectSynset(word, sentence):

    return synsetWord


def hypernyms_relation(words):
    hypernyms_list = []
    for word in words:
        try:
            w = wn.synsets(word)[0]
            hypernyms = w.hypernyms()
        except:
            w = "N/A"
            hypernyms = ["N/A"]
        hypernyms_list.extend(hypernyms)
    return hypernyms_list
def hyponyms_relation(words):
    hyponyms_list = []
    for word in words:
        try:
            w = wn.synsets(word)[0]
            hyponyms = w.hyponyms()
        except:
            w = "N/A"
            hyponyms = "N/A"
        hyponyms_list.extend(hyponyms)
    return hyponyms_list
def holonyms_relation(words):
    holonyms_list = []
    for word in words:
        try:
            w = wn.synsets(word)[0]
            holonyms = w.part_holonyms()
        except:
            w = "N/A"
            holonyms = "N/A"
        holonyms_list.extend(holonyms)
    return holonyms_list
def meronyms_relation(words):
    meronyms_list = []
    for word in words:
        try:
            w = wn.synsets(word)[0]
            meronyms = w.part_meronyms()
        except:
            w = "N/A"
            meronyms = "N/A"
        meronyms_list.extend(meronyms)
    return meronyms_list

def hypernyms_ratio(s1,s2):
    # s1_tokenized = tokenize(s1)
    # s2_tokenize = tokenize(s2)
    s1_hyper = hypernyms_relation(s1)
    s2_hyper = hypernyms_relation(s2)
    # print(s1_hyper)

    overlap_set = (set(s1_hyper)).intersection(set(s2_hyper))
    ratio = 0
    if (len(s1_hyper)+len(s2_hyper)) != 0:
        ratio = len(overlap_set)/(len(s1_hyper)+len(s2_hyper))
    return ratio

def hyponyms_ratio(s1,s2):
    # s1_tokenized = tokenize(s1)
    # s2_tokenize = tokenize(s2)
    s1_hypo = hyponyms_relation(s1)
    s2_hypo = hyponyms_relation(s2)
    overlap_set = set(s1_hypo).intersection(set(s2_hypo))
    ratio = 0
    if (len(s1_hypo)+len(s2_hypo)) != 0:
        ratio = len(overlap_set)/(len(s1_hypo)+len(s2_hypo))
    return ratio

def holonyms_ratio(s1,s2):
    # s1_tokenized = tokenize(s1)
    # s2_tokenize = tokenize(s2)
    s1_holo = holonyms_relation(s1)
    s2_holo = holonyms_relation(s2)
    overlap_set = set(s1_holo).intersection(set(s2_holo))
    ratio = 0
    if (len(s1_holo)+len(s2_holo)) != 0:
        ratio = len(overlap_set)/(len(s1_holo)+len(s2_holo))
    return ratio

def meronyms_ratio(s1,s2):
    # s1_tokenized = tokenize(s1)
    # s2_tokenize = tokenize(s2)
    s1_meronyms = meronyms_relation(s1)
    s2_meronyms = meronyms_relation(s2)
    overlap_set = set(s1_meronyms).intersection(set(s2_meronyms))
    ratio = 0
    if (len(s1_meronyms)+len(s2_meronyms)) != 0:
        ratio = len(overlap_set)/(len(s1_meronyms)+len(s2_meronyms))
    return ratio

