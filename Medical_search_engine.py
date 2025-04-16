import streamlit as st
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from io import StringIO
import plotly.graph_objects as go



# Command to make the display not truncated
pd.set_option("display.max_colwidth", -1)


# Function to find the cosine similarity
def cos_sim(a,b):
    return round(dot(a,b)/(norm(a)*norm(b)),3)


# Functions for text Preprocessing
# function to remove all urls 
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^A-Z0-9a-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

# make text lower case
def lower_case(text):
    text = text.lower()
    return text

# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+','',text)
    return result

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('','',string.punctuation)
    text = text.translate(translator)
    return text

# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words]
    return text

# lemmatize text
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Unifying all the above functions
def preprocessing(text):

    text = lower_case(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    
    return text

# Let the vector size be 100 dims.
vector_size = 100

# Load the 100dim skipgram and fasttext models
skipgram_100 = Word2Vec.load("skipgram_100.bin")
fast_n_100 = Word2Vec.load("fast_n_100.bin")


# Function to embedd the query with the choosen model
def get_mean_vector(model,words):
    words = [word for word in word_tokenize(words) if word in list(model.wv.index_to_key)] #Add the word to list of words if found in vocab
    if len(words)>=1:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.zeros([vector_size])
    

# Function to make the querry processed and extract embeddings for the querry
def preprocessing_input(query,model):
    query = preprocessing(query)
    query=query.replace('\n',' ')
    k = get_mean_vector(model,query)
    return k

# Now read the data and load it in a dataframe
with open("dimensions-covid19-export-2021-09-01-h15-01-02_clinical_trials.csv","r",encoding="utf-8",errors="ignore") as f:
    content = f.read()

df = pd.read_csv(StringIO(content))

df1 = df[["Date added","Trial ID","Title","Brief title","Abstract"]]



# Average vector of abstract
# in skipgram
k = pd.read_csv("method2_k_skipgram_100.csv")
skipgram_100_vectors = []
for i in range(df1.shape[0]):
    skipgram_100_vectors.append(k[str(i)].values)

# in fasttext
k1 = pd.read_csv("k_fasttext_abstract.csv")
fasttext_100_vectors = []
for i in range(df1.shape[0]):
    fasttext_100_vectors.append(k1[str(i)].values)


# Streamlit function
def main():
    # Load the data and models
    data = df1

    st.title("Clinical Trial Search Engine")
    st.write("Select Model")

    Vector_models = st.selectbox("Model",options=["skipgram_100","fast_n_100"])
    if Vector_models == "skipgram_100":
        K = skipgram_100_vectors
        word2vec = skipgram_100
    else:
        K = fasttext_100_vectors
        word2vec = fast_n_100

    
    st.write("Write your query")
    query = st.text_input("Search_box") #getting input from user

    #function to preprocess the input querry text
    def preprocessing_input(query,word2vec):
        query = preprocessing(query)
        query=query.replace('\n',' ')
        k = get_mean_vector(word2vec,query)
        return k
        
    # Function to find top n similar results
    def top_n(query,p,df1,word2vec):

        query = preprocessing_input(query,word2vec) #preprocessing the input

        x=[]
        # calculate the cosine similarities of input query with all the vector abstracts
        for i in range(len(p)):
            x.append(cos_sim(query,p[i]))

        temp = list(x)

        # sort the list to find the top n similar results
        res = sorted(range(len(x)),key=lambda sub: x[sub])[-5:] #This code gets the position of the highest similarity values

        simi = [temp[i] for i in reversed(res)] #This line is to get the similarity values based on the position
        print(simi)

        L = []
        for i in reversed(res):
            L.append(i)

        return df1.iloc[L,:],simi
    
    model = top_n

    if query:
        p,sim = model(str(query),K,df1,word2vec)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Date added','Trial ID','Title','Brief title','Abstract','Score'],
                align='left'
            ),
            cells=dict(
                values=[list(p['Date added'].values),list(p['Trial ID'].values),list(p['Title'].values),list(p['Brief title'].values),list(p['Abstract'].values),list(np.around(sim,3))],
                align='left'
            )
        )])
        # displaying our plotly table
        fig.update_layout(height=1700,width=700,margin=dict(l=0, r=10, t=20, b=20))

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
