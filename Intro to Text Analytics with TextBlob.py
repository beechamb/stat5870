## Import Packages ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Introduction to Text Analytics with TextBlob

# Install TextBlob and download the necessary NLTK corpora.
# !pip install -U textblob
# !python -m textblob.download_corpora

# Read the dataset.
df = pd.read_csv("amazon_cells_labelled.txt", names=['sentence', 'label'], sep="\t")
df.head()

df["sentence"][0]
df["sentence"][1]
df["sentence"][101]
df["sentence"][103]
df["sentence"][302]
df["sentence"][503]


# Tokenization
# Tokenization refers to splitting a large paragraph into sentences or words. 
# Typically, a token refers to a word in a text document. 
# Tokenization is pretty straight forward with TextBlob.
review = df["sentence"][101]

from textblob import TextBlob
textblob_obj = TextBlob(review)

textblob_obj.words
len(textblob_obj.words)

textblob_obj.sentences
len(textblob_obj.sentences)


# Lemmatization
# Lemmatization refers to reducing the word to its root form as found in a dictionary.
# ADJ, ADV, NOUN, VERB = 'a', 'r', 'n', 'v'
from textblob import Word

word1 = Word("apples")
word1.lemmatize("n")

word2 = Word("studied")
word2.lemmatize("v")

word3 = Word("greater")
word3.lemmatize("a")


# Word counts
# To find the frequency of occurrence of a particular word.
review = df["sentence"][103]
textblob_obj2 = TextBlob(review)

textblob_obj2.word_counts["good"]


# N-Grams
# N-Grams refer to N combination of words in a sentence. 
# N-Grams can play a crucial role in text classification.
textblob_obj.ngrams(2)


# Language Translation
textblob_obj_korean = TextBlob('감사합니다')
textblob_obj_korean.translate(to='en')

textblob_obj_chinese = TextBlob('谢谢')
textblob_obj_chinese.translate(to='en')

textblob_obj_english = TextBlob('Thank you')
textblob_obj_english.translate(to='ar')

textblob_obj_english = TextBlob('Thank you')
textblob_obj_english.translate(to='es')

textblob_obj_spanish = TextBlob('Gracias')
textblob_obj_spanish.translate(to='fr')

unknown = TextBlob('आपका धन्यवाद')
unknown.detect_language()


# Sentiment Analysis
# Polarity can be between -1 and 1, 
# 1 means positive statement and -1 means a negative statement.

# Subjectivity can be between 0 and 1,
# 1 means subjective statement and 0 means objective statement.

df["sentence"][1]
df["sentence"][101]
df["sentence"][103]

textblob_obj = TextBlob(df["sentence"][1])
textblob_obj.sentiment

textblob_obj = TextBlob(df["sentence"][101])
textblob_obj.sentiment

textblob_obj = TextBlob(df["sentence"][103])
textblob_obj.sentiment

def find_pol(review):
    return TextBlob(review).sentiment.polarity

df['Sentiment_Polarity'] = df["sentence"].apply(find_pol)
df.head()

sns.distplot(df["Sentiment_Polarity"])

most_negative = df[df["Sentiment_Polarity"] == -1]
most_negative["sentence"]

