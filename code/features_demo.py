def wordnetFeatures(words):
    hypernyms_list = []
    hyponyms_list = []
    holonyms_list = []
    meronyms_list = []
    for word in words:
        try:
            w = wn.synsets(word)[0]
            hypernyms = w.hypernyms()
            hyponyms = w.hyponyms()
            holonyms = w.part_holonyms()
            meronyms = w.part_meronyms()
        except:
            w = "N/A"
            hypernyms = "N/A"
            hyponyms = "N/A"
            holonyms = "N/A"
            meronyms = "N/A"
        # print("The word is ",word,". 1. ",w," 2. ",hypernyms," 3. ",hyponyms," 4. ",holonyms," 5. ",meronyms)   
        hypernyms_list.append(hypernyms)
        hyponyms_list.append(hyponyms)
        holonyms_list.append(holonyms)
        meronyms_list.append(meronyms)
    return (hypernyms_list, hyponyms_list, holonyms_list, meronyms_list)