import numpy                  #used in tensoerflow for training ,used for some array management
import random                 #for chooice a random answer from dataset
from time import sleep        

# chat function will handle getting a prediction from the model and grabbing an appropriate response
#  from  our JSON file of responses.
def chat(model,bag_of_words,labels,words,data):  
    
    print("Hi, How can i help you ?")       #starting message
    while True:                             #while asking msg
        inp = input("You: ")                #user message or question preceded by you:
        if inp.lower() == "quit":           #if user type quit the chat with bot will be ended
            break

        results = model.predict([bag_of_words(inp, words)]) #make a prediction from the model according to bag_of_words function
        results_index = numpy.argmax(results)                  #convert results to numpy array 
        tag = labels[results_index]
        
        if results[results_index] > 0.8:         #if the result or prediction more than 80 % ,The prediction is displayed as an output on the chat utility window as a response from the bot
            for tg in data["intents"]:           
                if tg['tag'] == tag:      
                    responses = tg['responses'] 
            sleep(1)                         # take 1 sec for response
            Bot = random.choice(responses)   #choice random response from dataset
            print(Bot)                       #print response 
        else:
            print("I don't understand!")     #if the user type any msg o that the bot not understand 