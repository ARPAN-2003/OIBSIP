import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()

    text = re.sub("[^a-zA-Z ]"," ",text)

    words = text.split()

    words = [ps.stem(word)
             for word in words
             if word not in stopwords.words("english")]

    return " ".join(words)