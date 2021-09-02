import nltk                                  #natural language toolkit
import json                                  #to read the json files (dataset ) that we need to train our model
import pickle                                #for saving data after shapping
import numpy                                 #used in tensoerflow for training,used for some array management

def data_preprocessing(stemmer):            #function for preprocessing data
    with open("intents.json") as file:         #Loading our JSON Data
        data = json.load(file)
     #Extracting Data
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:                                     #looping in json data (looping through all dictionaries such patterns and all of that)
        words = []                              # put the words we are whant to tokonize
        labels = []                             #store tags in it
        docs_x = []                             #put the wordes without tokonization
        docs_y = []                             #store the relationship between wordes and tags
        punctuation=["?","!"," "," .",",","&"]  #ignore this punctuation

                                                #Now its time to loop through our JSON data and extract the data we want. 
                                                # For each pattern we will turn it into a list of words using nltk.
                                                # word_tokenizer, rather than having them as strings.
                                                #access the patterns 
                                                # Now we start stemming
                                                #take each words in our patterns and bring it down to the root word
                                                #eliminating extra charcters
                                                #ignore things that could stray the model in the wrong direction
                                                #example intent (anyone there?) the root of there? is "there" and whats up will be what
        for intent in data ["intents"]:

            for pattern in intent["patterns"]:
                                                     #put all of these tokenized words into our words list
                wrds = nltk.word_tokenize(pattern)
                                                      #from previous step that doing looping this step will appending each one 
                                                      #so we will add all words in
                words.extend(wrds)
                                                       #  We will then add each pattern into our docs_x list and its associated tag into the docs_y list
                                                       #for each pattern in docs-X we want to put another element in docs-Y that stands for what intent it's a part of 
                                                       #each entry in docs-X corresponds to an entry in docs-Y
                                                       #the entry in Docs-X is going to be that pattern and the intent will be in docs-Y
                                                       #so we know kind of how to classify each of our patterns which will be important for training model
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
                                                       #append all different tags to labels
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                                                      #Word Stemming: Stemming a word is attempting to find the root of the word
                                                    #We will use this process of stemming words 
                                                    # to reduce the vocabulary of our model and attempt to find the more general meaning behind sentences.
                                                    #stem all of the words that we have in words list And Remove any duplicate elements
                                                
                                                    #convert all of our words into lowercase  
    
        words = [stemmer.stem(w.lower()) for w in words if w != punctuation]
                                                     #set => it takes all the words make sure there's no duplicates or just remove any duplicate elements list 
                                                    #and convert this back into a list and store it obviously 
                                                    #and sort these words
        words = sorted(list(set(words)))
    
        labels = sorted(labels)                          #sort our labels

                                                         #Bag of Words (# if work exests append 1 else append 0)
                                                         #creating training testing output
        training = []
        output = []
    

#  Now all is done essentially set up these few lists so we have all of our labels in one list,
#  all of the different words in our patterns in one , we have docs-x which has list of all of the different 
#  patterns and then docs-Y and the corresponding entries in docs-X and docs-Y are like the words and then tag for 
#  those words which is the pattern 
#  model more accurate

        
        out_empty = [0 for _ in range(len(labels))]
                                                            #that is a bunch of output list they are all going to be the length of the amount of tags we have 
                                                            #Put 0 for all labels or tags 
                                                            #if the calss or tag exist that we want put 1 
        for x, doc in enumerate(docs_x):
            bag = []
    
            wrds = [stemmer.stem(w) for w in doc]     
    
            for w in words:
                if w in wrds:
                    bag.append(1)                  #put 1 if the words already exists 
                else:
                    bag.append(0)                  #else put 0 if the words dose not exist
                                                                # [1,1,1,0]
                                                                 # 'greeting','goodbye','age','what'
                                                                 #meaning that the word ' what ' is exist so it put 0 and vice versa
    
    
    
    
    
        training = numpy.array(training)          #convert training to numpy array
        output = numpy.array(output)              #convert output to numpy array
    
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
        
        return training, output,labels,words,data
