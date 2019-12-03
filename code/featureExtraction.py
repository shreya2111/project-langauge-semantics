import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

import features
import wordnetFeatures as wordnetFeatures
import similarity_features as similarityFeatures
import dependency_parse as dependencyParseTreeFeatures

def extract_features(s1,s2):
    # featureVector = pd.DataFrame(
    #     columns = ['Jaccard','bow','cosine','hyper_ratio',
    #     'hypo_ratio','holo_ratio','mero_ratio']
    #     )
    featureVector = []
    s1_tokenized  = features.tokenize(s1)
    s2_tokenized  = features.tokenize(s2)
    s1_lemmatized = features.lemmatize(s1_tokenized)
    s2_lemmatized = features.lemmatize(s2_tokenized)
    
    # similarity and bow features
    s1_lemmatized_two = " ".join(s1_lemmatized)
    s2_lemmatized_two = " ".join(s2_lemmatized)
    jaccardSimilarity = similarityFeatures.jaccard_similarity(s1_lemmatized_two,s2_lemmatized_two)
    bow               = similarityFeatures.bow(s1_lemmatized,s2_lemmatized)
    cosine_similarity = similarityFeatures.cosine_similarity(s1_tokenized,s2_tokenized)

    # wordnet ratio features
    ratio_hyper = wordnetFeatures.hypernyms_ratio(s1_tokenized,s2_tokenized)
    ratio_hypo  = wordnetFeatures.hyponyms_ratio(s1_tokenized,s2_tokenized)
    ratio_holo  = wordnetFeatures.holonyms_ratio(s1_tokenized,s2_tokenized)
    ratio_mero  = wordnetFeatures.meronyms_ratio(s1_tokenized,s2_tokenized)

    # depedency parse features


    # # pos tag features
    # s1_pos_tags = pos_tag(s1_lemmatized)
    # s2_pos_tags = pos_tag(s2_lemmatized)
    
    # Putting all features together
    # featureVector.append(jaccardSimilarity)
    zipFeatures = [jaccardSimilarity,bow,cosine_similarity,ratio_hyper,ratio_hypo,ratio_holo,ratio_mero]
    featureVector.extend(zipFeatures)

    return featureVector