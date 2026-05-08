import re
import string
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import sys
import contractions
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# 1) Text Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove html tags
    return text

text_file = "C:/Users/User/OneDrive/Documents/ResearchProjectText.csv"
text_pd = pd.read_csv(text_file, index_col=0)
text_transcript = text_pd["Transcript Text"]
text_transcript_list = text_transcript.tolist()
text_transcript_list = [str(doc) for doc in text_transcript_list]
cleaned_text_transcript_list = [clean_text(doc) for doc in text_transcript_list]

# 2) Contractions
expanded_corpus = [contractions.fix(doc) for doc in cleaned_text_transcript_list]

# 3) Tokenization
tokenized_corpus = [word_tokenize(doc) for doc in expanded_corpus]

# 4) Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_corpus = [[word for word in doc if word not in stop_words] for doc in tokenized_corpus]

# 5) Stemming and Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_corpus = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_corpus]
text_lemmatized_corpus = [" ".join(words) for words in lemmatized_corpus]

with open("C:/Users/User/Downloads/Text_Preprocessing.txt", "w") as f:
    f.write(str(text_lemmatized_corpus))




