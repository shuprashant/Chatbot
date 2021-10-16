import nltk 
import io
import numpy as np
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
f=open("E:\SEM 7\AI\project\covid.txt",'r',errors='ignore')
raw=f.read()
raw=raw.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens=nltk.sent_tokenize(raw)
word_tokens=nltk.word_tokenize(raw)
learner=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [learner.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
GREETING_INPUTS=("hello","hi","greetings","sup","what's up","hey")
GREETING_RESPONSES=['hi','hey',"*nodes*","hi there",'i am glad! you are talking to me']
def greetings(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
def responses(user_response):
    chatbot_response=''
    sent_tokens.append(user_response)
    TfidVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfid=TfidVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfid[-1],tfid)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfid=flat[-2]
    if (req_tfid==0):
        chatbot_response=chatbot_response+"I am sorry! I don't Understand You"
        return chatbot_response
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]
        return chatbot_response
flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input("USER:")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greetings(user_response)!=None):
                print("ROBO: "+greetings(user_response))
            else:
                print("ROBO: ",end="")
                print(responses(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


