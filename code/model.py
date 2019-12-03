import sys
from sklearn import svm
import featureExtraction 
from readData import CorpusReader
from sklearn import svm

# class model:


if __name__=="__main__":
#     trainSet = sys.argv[1]    
#     datasetType = sys.argv[2]
    s1 = "The young lady enjoys listening to the guitar."
    s2 = "The young lady enjoys playing the guitar."
    tag = 4

    # extract features
    features = featureExtraction.extract_features(s1,s2)
    dataset  = pd.Dataframe(features, columns = ['Jaccard','bow','cosine','hyper_ratio',
        'hypo_ratio','holo_ratio','mero_ratio'])
    label    = pd.Dataframe(tag, column = ['gold_tag'])

    # call model and save model
    # evaluate