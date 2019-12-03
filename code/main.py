import sys
import nltk
import pandas as pd
import featureExtraction
from sklearn import svm
from nltk import wordnet as wn
import evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


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

    def readThroughPandas(self):
        df = pd.read_csv(self.path, delimiter = "\t+", engine = 'python')
        # print(df.head(10))
        return df

    def getDataset(self,df,datasetType):
        features = []
        tag    = []
        for index,row in df.iterrows():
            if datasetType is 'test':
                # data.append(features.extract_features(row['Sentence1'],row['Sentence2'],row['GoldTag']))
                features.append(featureExtraction.extract_features(
                    row['Sentence1'], row['Sentence2'])
                    )
            else:
                features.append(
                    featureExtraction.extract_features(
                    row['Sentence1'], row['Sentence2']
                    ))
                tag.append(row['GoldTag'])
        if datasetType is 'train' or 'dev':
            label   = pd.DataFrame(tag, columns = ['gold_tag'])
        dataset = pd.DataFrame(features, columns = ['Jaccard','bow','cosine','hyper_ratio',
            'hypo_ratio','holo_ratio','mero_ratio'])
        return dataset, label

if __name__=="__main__":
    train_set = sys.argv[1]
    dev_set = sys.argv[2]
    # datasetType = sys.argv[3]

    datasetType = 'train'
    trainData = CorpusReader(train_set)
    devData   = CorpusReader(dev_set)  
    
    df_train = trainData.readThroughPandas()
    df_dev   = devData.readThroughPandas()

    # extract features
    train_dataset, train_label = trainData.getDataset(df_train,datasetType)
    # print(train_label.head(10))
    dev_dataset, dev_label = devData.getDataset(df_dev, datasetType)

    # call model and save model

    # TO DO: func call for model
    classifier = svm.SVC(gamma = 'scale')
    classifier.fit(train_dataset, train_label.values.ravel())

    # func call for predict
    dev_pred = classifier.predict(dev_dataset)

    # checking accuracy & error
    devSet_accuracy = accuracy_score(dev_label, dev_pred)
    # tn, fp, fn, tp = confusion_matrix(dev_label, dev_pred).ravel()
    print(devSet_accuracy)

    # print(confusion_matrix(dev_label, dev_pred).ravel())
    # print("Confusion matrix: ", tn, fp, fn, tp)
    

    # evaluate
    print(type(dev_label))
    correlation_score = evaluation.get_correlation(dev_label.to_dict(), dev_pred.to_dict())
    print("\nPearson correlation coefficient: ", correlation_score)
