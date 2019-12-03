import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import wordnetFeatures as wordnetFeatures
import similarity_features as similarityFeatures
import dependency_parse as dependencyParseTreeFeatures

def lemmatize(s):
    lemmatizer = WordNetLemmatizer()
    l = []
    for i in s:
        l.append(lemmatizer.lemmatize(i))
    return(l)
    
def pos_tag(s):
    l = nltk.pos_tag(s)
    return(l)

def tokenize(d):
    l = d.split(" ")
    for i in range(len(l)):
        l[i] = l[i].lower()
    s = " ".join(l)
    l = nltk.word_tokenize(s)
    return(l)

def ner_overlap(self,s1,s2):
    sent1=nlp(s1)
    sent2=nlp(s2)
    ne_sent1=[x.label_ for x in sent1.ents]
    ne_sent2=[x.label_ for x in sent2.ents]
    print(ne_sent1,ne_sent2)        
    overlap=0
    if(ne_sent1==[] or ne_sent2==[]):
        return(0)
    else:
        overlap= (2*len(set(ne_sent1).intersection(set(ne_sent2))))/(len(ne_sent1)+ len(ne_sent2))
        return(overlap)
    
def action_verb(self,s1,s2):
    verb_s1=[]
    verb_s2=[]
    for i in s1:
        s=self.pos_tag([i])
        if(s[0][1] in ['VB','VBD','VBG','VBN','VBP','VBZ']):
            verb_s1.append(i)
    for i in s2:
        p=self.pos_tag([i])
        if(p[0][1] in ['VB','VBD','VBG','VBN','VBP','VBZ']):
            verb_s2.append(i)
    #print(verb_s1,verb_s2)
    action_verb_overlap=0
    action_verb_overlap=2*(len(set(verb_s1).intersection(set(verb_s2))))/(len(verb_s1)+len(verb_s2))
    return (action_verb_overlap)
        


