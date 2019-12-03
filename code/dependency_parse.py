# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sys
import pandas as pd
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import spacy
import re
nlp = spacy.load("en_core_web_sm")


def dependency_parse(self,s):
    #l=" ".join(s)
    doc = nlp(s)
    for token in doc:
        print(token.text, token.tag_, token.head.text, token.dep_)

def dependency_parse_features(self,s1,s2):
    sent1=nlp(s1)
    sent2=nlp(s2)
    head1=""
    head2=""
    obj1=""
    obj2=""
    subj1=""
    subj2=""
    # finding head, subj, obj for sentence 1
    for token in sent1:
        if(token.dep_=="ROOT"):
            head1=token.text
        if(token.dep_=="dobj"or token.dep_=="pobj" or token.dep_=="obj"):
            obj1=token.text
        if(token.dep_=="nsubj"):
            subj1=token.text
            
    # finding head, subj, obj for sentence 2
    for token in sent2:
        if(token.dep_=="ROOT"):
            head2=token.text
        if(token.dep_=="dobj" or token.dep_=="pobj" or token.dep_=="obj"):
            obj2=token.text
        if(token.dep_=="nsubj"):
            subj2=token.text
    print(head2,subj2,obj2)
    
    # find 1st synset of head1
    if(wn.synsets(head1)!=[]):
        h1_synset=wn.synsets(head1)[0]
    else:
        h1_synset=[]
    if(wn.synsets(subj1)!=[]):
        sub1_synset=wn.synsets(subj1)[0]
    else:
        sub1_synset=[]
    if(wn.synsets(obj1)!=[]):
        obj1_synset=wn.synsets(obj1)[0]
        print(obj1_synset)
    else:
        obj1_synset=[]
    # sent 2 synsets
    if(wn.synsets(head2)!=[]):
        h2_synset=wn.synsets(head2)[0]
    else:
        h2_synset=[]
    if(wn.synsets(subj2)):
        sub2_synset=wn.synsets(subj2)[0]
    else:
        sub2_synset=[]
    if(wn.synsets(obj2)!=[]):
        obj2_synset=wn.synsets(obj2)[0]
    else:
        obj2_synset=[]
    
    #hypernym sent1
    if(str(h1_synset)!=""):
        head1_hypernym=h1_synset.hyponyms()
    else:
        head1_hypernym=[]
    if(str(sub1_synset)!=""):
        subj1_hypernym=sub1_synset.hyponyms()
    else:
        subj1_hypernym=[]
    if(str(obj1_synset)!=""):
        obj1_hypernym=obj1_synset.hyponyms()
    else:
        obj1_hypernym=[]
        
    # hypernym sent2
    if(str(h2_synset)!=""):
        head2_hypernym=h2_synset.hyponyms()
    else:
        head2_hypernym=[]
    if(str(sub2_synset)!=""):
        subj2_hypernym=sub2_synset.hyponyms()
    else:
        subj2_hypernym=[]
    if(str(obj2_synset)!=""):
        obj2_hypernym=obj2_synset.hyponyms()
    else:
        obj2_hypernym=[]
    print("sent1",head1,subj1,obj1)
    print("hypernym sent 1:",head1_hypernym,subj1_hypernym,obj1_hypernym)
    print("sent1",head2,subj2,obj2)
    print("hypernym sent 2:",head2_hypernym,subj2_hypernym,obj2_hypernym)
    print("Head 1 and Head 2 intersection")
    print("******")   
    print(len(head1_hypernym))
    # print(head1_hypernym)
    # print(head2_hypernym)
    #print(set(head1_hypernym).intersection(set(head2_hypernym)))
    head_overlap=len((set(head1_hypernym).intersection(set(head2_hypernym))))/(len(head1_hypernym)+len(head2_hypernym))
    subj_overlap=len((set(subj1_hypernym).intersection(set(subj2_hypernym))))/(len(subj1_hypernym)+len(subj2_hypernym))
    obj_overlap=len((set(obj1_hypernym).intersection(set(obj2_hypernym))))/(len(obj1_hypernym)+len(obj2_hypernym))
    print("ratios")
    print(head_overlap)
    print(subj_overlap)
    print(obj_overlap)


def dparse_comparison(self,s1,s2):
    sent1=nlp(s1)
    sent2=nlp(s2)
    head1=""
    head2=""
    obj1=""
    obj2=""
    subj1=""
    subj2=""
    # finding head, subj, obj for sentence 1
    for token in sent1:
        if(token.dep_=="ROOT"):
            head1=token.text
        if(token.dep_=="dobj"or token.dep_=="pobj" or token.dep_=="obj"):
            obj1=token.text
        if(token.dep_=="nsubj"):
            subj1=token.text
            
    # finding head, subj, obj for sentence 2
    for token in sent2:
        if(token.dep_=="ROOT"):
            head2=token.text
        if(token.dep_=="dobj" or token.dep_=="pobj" or token.dep_=="obj"):
            obj2=token.text
        if(token.dep_=="nsubj"):
            subj2=token.text
    
    vect=[]
    if(head1==head2):
        vect.append(1)
    else:
        vect.append(0)
    if(obj1==obj2):
        vect.append(1)
    else:
        vect.append(0)
    if(subj1==subj2):
        vect.append(1)
    else:
        vect.append(0)
    
    print("vector:",vect)
        
    
    
    def jaccard_similarity(self,query, document):
        query=query.split(" ")
        document=document.split(" ")
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        print(intersection)
        print(union)
        return len(intersection)/len(union)
    
    def resnik_similarity(self,s1,s2):
        s1=self.tokenize(s1)
        s2=self.tokenize(s2)
        # TO-DO    
                
    def bow(self,s1,s2):
        s1=self.tokenize(s1)
        s2=self.tokenize(s2)
        s1=self.lemmatize(s1)
        s2=self.lemmatize(s2)
        print(s1,s2)
        overlap=0
        normalized_score=0
        N= len(s1)+len(s2)
        overlap=len(set(s1).intersection(set(s2)))
        normalized_score=overlap/N
        return(normalized_score)
        
    
    def cosine_similarity(self,s1,s2):
        s1=self.tokenize(s1)
        s2=self.tokenize(s2)
        # remove stop words
        sw=stopwords.words('english')
        s1 = {w for w in s1 if w not in sw}  
        s2 = {w for w in s2 if w not in sw} 
        l1=[]
        l2=[]
        print(s1,s2)
        rvector=set(s1).union(set(s2))
        #rv=list(rvector)
        for w in rvector:
            if(w in s1):
                l1.append(1)
            else:
                l1.append(0)
            if(w in s2):
                l2.append(1)
            else:
                l2.append(0)
        c = 0
      # cosine formula  
        for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        print("similarity: ", cosine) 
        print("s1 vector:",l1)
        print("s2 vector:",l2)
            
        
        

if __name__=="__main__":
    train_set = open("C:\\Users\\apurv\\Desktop\\data\\data\\train-set.txt",encoding='utf8')
    
    #df = pd.read_csv("C:\\Users\\apurv\\Desktop\\data\\data\\train-set.txt",delimiter="\t+")
    # dev_set = sys.argv[2]

    corpusData = CorpusReader(train_set)
    # corpusData.readData()
    #corpusData.readThroughPandas()
    
    l=corpusData.tokenize("Micron's has declared its first quarterly profit for three years.")
    s=corpusData.lemmatize(l)
    corpusData.pos_tag(s)
    # pass the lemmatized sentence to dependency parser
    corpusData.dependency_parse(s)
    corpusData.synsets(l)
    
    ## Testing JACCARD SIMILARITY
    
    s1="The young lady enjoys listening to the guitar."
    s2="The young lady enjoys playing the guitar."
    s1_tokenize=corpusData.tokenize(s1)
    s1_lemmatize=corpusData.lemmatize(s1_tokenize)
    s1_lemmatize=" ".join(s1_lemmatize)
    s2_tokenize=corpusData.tokenize(s2)
    s2_lemmatize=corpusData.lemmatize(s2_tokenize)
    s2_lemmatize=" ".join(s2_lemmatize)
    print(corpusData.jaccard_similarity(s1_lemmatize,s2_lemmatize))
    corpusData.cosine_similarity(s1,s2)
    print("BOW score:",corpusData.bow(s1,s2))
    corpusData.dependency_parse(s2)
    corpusData.dependency_parse_features(s1,s2)
    corpusData.dparse_comparison(s1,s2)
    # resnik similarity- with wordnet senses
    
    
    
    
    
    
        


    
