from nltk.corpus import stopwords

def jaccard_similarity(query, document):
        query    = query.split(" ")
        document = document.split(" ")
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        # print(intersection)
        # print(union)
        return len(intersection)/len(union)
    
def resnik_similarity(s1,s2):
    s1 = self.tokenize(s1)
    s2 = self.tokenize(s2)
    # TO-DO    
            
def bow(s1,s2):
    # s1=self.tokenize(s1)
    # s2=self.tokenize(s2)
    # s1=self.lemmatize(s1)
    # s2=self.lemmatize(s2)
    # print(s1,s2)
    overlap=0
    normalized_score=0
    N= len(s1)+len(s2)
    overlap=len(set(s1).intersection(set(s2)))
    normalized_score=overlap/N
    return normalized_score
    
def cosine_similarity(s1,s2):
    # s1=self.tokenize(s1)
    # s2=self.tokenize(s2)
    # remove stop words
    sw = stopwords.words('english')
    s1 = {w for w in s1 if w not in sw}  
    s2 = {w for w in s2 if w not in sw} 
    l1=[]
    l2=[]
    # print(s1,s2)
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
    # print("similarity: ", cosine) 
    # print("s1 vector:",l1)
    # print("s2 vector:",l2)
    return cosine
            
