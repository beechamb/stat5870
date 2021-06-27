#Importing Necessary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import string
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Importing Datasets
trump = pd.DataFrame(pd.read_csv("trumptweets.csv"))
biden = pd.read_csv("JoeBidenTweets.csv")

#make biden's tweets from 2009 onward to match trump's
biden = biden.loc[(biden["timestamp"] >= "2009-01-01 00:00")]

#Getting a feel for the data
trump.head()
biden.head()

trump.columns
biden.columns

#how many tweets did each user compose?
trump.shape[0]
biden.shape[0]

#SENTIMENT ANALYSIS
#Function for analyzing sentiment
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"
    
#Adding columns to both dataframes with the sentiment analysis
trump["sentiment"] = np.array([analyze_sentiment(tweet) for tweet in trump["content"]])
biden["sentiment"] = np.array([analyze_sentiment(tweet) for tweet in biden["tweet"]])

#Countplots of sentiments
#trump
sns.countplot(x = "sentiment", data = trump).set(title="Sentiment Counts of Trump's Tweets")

#biden
sns.countplot(x = "sentiment", data = biden).set(title="Sentiment Counts of Biden's Tweets")

#Proportions of positive, negative and neutral tweets for each user
positivetrump = (np.sum(trump["sentiment"] == "positive"))/(trump.shape[0])
positivetrump

positivebiden = (np.sum(biden["sentiment"] == "positive"))/(biden.shape[0])
positivebiden

negativetrump = (np.sum(trump["sentiment"] == "negative"))/(trump.shape[0])
negativetrump

negativebiden = (np.sum(biden["sentiment"] == "negative"))/(biden.shape[0])
negativebiden

neutraltrump = (np.sum(trump["sentiment"] == "neutral"))/(trump.shape[0])
neutraltrump

neutralbiden = (np.sum(biden["sentiment"] == "neutral"))/(biden.shape[0])
neutralbiden


#WORD FREQUENCY
#punctuation that must be removed, along with a few other necessary removals/custom "stop words"
punctuation = ["+",",",".","-","'","\"","&","!","?",":",";","#","~","=","/","$",
               "£","^","(",")","_","<",">", "--", "--", "—", "@","%", "``", "''",
               "’", "http", "'s", "…", "”", "“", "https", "n't", "get", "realdonaldtrump"]
stop_words = stopwords.words('english')

#function for removing html
def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    html_free = soup.get_text()
    return html_free

#function for removing punctuation
def remove_punctuation(text):
    no_punct = [w for w in text if w not in punctuation]
    return no_punct




#REALDONALDTRUMP
#combining and tokenizing all of trump's tweets
trumptweets= []
trumpwords = []
for tweet in trump["content"]:
    trumptweets.append(word_tokenize(tweet))
    
#removing stop words for trump's tweets
for tweet in trumptweets:
    for word in tweet:
        word = word.lower()
        if word not in stop_words:
            trumpwords.append(word)

#removing punctuation
trumpwords = remove_punctuation(trumpwords)

#counting most frequent words used by donald trump
trumpcount = Counter(trumpwords)
trumpmostcommon = trumpcount.most_common(10)
trumpmostcommon


#plot of trump's most common words
dftrump = pd.DataFrame(trumpmostcommon, columns = ["word", "count"])

plt.bar(range(len(dftrump["word"])), dftrump["count"])
plt.xticks(range(len(dftrump["word"])), dftrump["word"], rotation = "vertical")
plt.title("Donald Trump's Most Frequently Tweeted Words")

#wordcloud of trump's most common words
trumpstring = ' '.join(trumpwords)
trumpstring = remove_html(trumpstring)

trumpwordcloud = WordCloud(max_words = 50).generate(trumpstring)
plt.figure()
plt.imshow(trumpwordcloud)

#JOE BIDEN
#combining and tokenizing all of biden's tweets
bidentweets= []
bidenwords = []
for tweet in biden["tweet"]:
    bidentweets.append(word_tokenize(tweet))
    
#removing stop words for biden's tweets
for tweet in bidentweets:
    for word in tweet:
        word = word.lower()
        if word not in stop_words:
            bidenwords.append(word)

#removing punctuation
bidenwords = remove_punctuation(bidenwords)

#counting most frequent words used by joe biden
bidencount = Counter(bidenwords)
bidenmostcommon = bidencount.most_common(10)
bidenmostcommon

#plot of biden's most common words
dfbiden = pd.DataFrame(bidenmostcommon, columns = ["word", "count"])

plt.bar(range(len(dfbiden["word"])), dfbiden["count"])
plt.xticks(range(len(dfbiden["word"])), dfbiden["word"], rotation = "vertical")
plt.title("Joe Biden's Most Frequently Tweeted Words")

#word cloud of biden's most common words
bidenstring = ' '.join(bidenwords)
bidenstring = remove_html(bidenstring)

bidenwordcloud = WordCloud(max_words = 50).generate(bidenstring)
plt.figure()
plt.imshow(bidenwordcloud)

#COLLOCATION
#trump
colloctrump = BigramCollocationFinder.from_words(trumpwords)
colloctrump.nbest(BigramAssocMeasures.likelihood_ratio, 5)

trumpbiwordcloud = WordCloud(stopwords = STOPWORDS,
                      collocation_threshold = 2).generate(trumpstring)
plt.imshow(trumpbiwordcloud, interpolation='bilinear')

#biden
collocbiden = BigramCollocationFinder.from_words(bidenwords)
collocbiden.nbest(BigramAssocMeasures.likelihood_ratio, 5)


#wordclouds
trumpwordcloud = WordCloud(stopwords = STOPWORDS,
                           collocations=True).generate(trumpwords)