# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#afile=open("C:\\Users\\apurv\\Desktop\\data\\data\\train-set.txt",encoding='utf8')

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
nlp = spacy.load(r'C:\Users\apurv\Anaconda3\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.2.0')



class CorpusReader:

    def __init__(self,dataset):
        self.path = dataset


    def readData(self):
        data = open(self.path,'r')
        # print(data)
        for line in data:
            # print(line)
            words = line.split("    ")
            # print(words)
            word_list = words[0].split("\t")
            # print(word_list[1]," ",word_list[2]," ",word_list[3]," ")
            # CorpusReader.tokenize(self,word_list[1])
            print(word_list)

    def readThroughPandas(self):
        df = pd.read_csv(self.path, delimiter = "\t+")
        print(df.head(10))
    
    def tokenize(self,d):
        #d['Sentence1']=d['Sentence1'].apply(lambda x: x.lower())
        #d['Tokenized_s1']=d['Sentence1'].apply(nltk.word_tokenize)
        #print(d['Tokenized_s1'])
        #l=d['Sentence1'].split(" ")
        l=d.split(" ")
        #print(l)
        for i in range(len(l)):
            l[i]=l[i].lower()
        s=" ".join(l)
        #print(s)
        #print(type(d['Sentence1']))
        l=nltk.word_tokenize(s)
        #print(l)
        return(l)
    
    def lemmatize(self,s):
        lemmatizer = WordNetLemmatizer()
        l=[]
        for i in s:
            l.append(lemmatizer.lemmatize(i))
        return(l)
    
    def pos_tag(self,s):
        #l=[]
        #for i in range(len(s)):
         #   l.append(nltk.pos_tag(s[i]))
        l=nltk.pos_tag(s)
        return(l)
     
    def dependency_parse(self,s):
        l=" ".join(s)
        doc = nlp(l)
        for token in doc:
            print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
    
    def synsets(self,s):
        for i in s:
            print(wn.synsets(i))
    
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
    
    s1="Birdie is washing itself in the water basin."
    s2="The bird is bathing in the sink."
    s1_tokenize=corpusData.tokenize(s1)
    s1_lemmatize=corpusData.lemmatize(s1_tokenize)
    s1_lemmatize=" ".join(s1_lemmatize)
    s2_tokenize=corpusData.tokenize(s2)
    s2_lemmatize=corpusData.lemmatize(s2_tokenize)
    s2_lemmatize=" ".join(s2_lemmatize)
    print(corpusData.jaccard_similarity(s1_lemmatize,s2_lemmatize))
    corpusData.cosine_similarity(s1,s2)
    print("BOW score:",corpusData.bow(s1,s2))
    
    # resnik similarity- with wordnet senses
    
    
    
    
    
    
        


    
