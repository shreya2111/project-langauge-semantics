import sys
import pandas as pd
import nltk

class CorpusReader:

    def __init__(self,dataset):
        self.path = dataset

    def tokenize(self,word_list):
        print("---TO DO---")

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

    def readThroughPandas(self):
        df = pd.read_csv(self.path, delimiter = "\t+")
        print(df.head(10))
        

if __name__=="__main__":
    train_set = sys.argv[1]
    # dev_set = sys.argv[2]

    corpusData = CorpusReader(train_set)
    # corpusData.readData()
    corpusData.readThroughPandas()
    
        