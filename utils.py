import pyarabic.araby as araby
import re
from nltk.corpus import stopwords
import emoji
import emosent
import pandas as pd
import numpy as np

def load_data(path: str) -> tuple:
    data = pd.read_csv(path)
    X = np.array(data["comment"])
    y = np.array(data["sentiment"])
    return (X, y)

def create_emoji_dict(comments: list) -> dict:
    """
    Create dictionary mappring emoji to sentiment (positive, negative)
    from text

    Parameters
    ----------
    comments : List
        list of comments containing emojis
    
    Returns
    -------
    out : Dictionary
        emoji_dict -- Dictionary containing emoji with corresponding sentiment
    """
    emoji_dict = {}
    emoji_set = set()
    for i in range(len(comments)):
        if emoji.distinct_emoji_list(comments[i]):
            emoji_set.update(emoji.distinct_emoji_list(comments[i]))
    for emo in emoji_set:
        if emosent.get_emoji_sentiment_rank(emo):
            emoji_dict[emo] = "positive" if emosent.get_emoji_sentiment_rank(emo)["sentiment_score"] > 0 else "negative"
    return emoji_dict

def normalize_emoji(comment: str, emoji_dict: dict) -> str:
    """ 
    Convert emoji to sentiment (positive, negative)

    Parameters
    ----------
    comment : String
        comment containing emojis
    emoji_dict : Dictionary
        dicitonary mapping emoji to sentiment (positive, negative)
    
    Returns
    -------
    out : String
        commnet -- String with normalized emojis
    """
    
    emoji_list = emoji_dict.keys()
    for e in emoji_list:
        comment = str.replace(comment, e, emoji_dict[e]+" ")

    return comment

def clean_text(comment: str) -> str:
    """
    Clean comment from noise.
    
    Parameters
    ----------
    comment : String
        comment to clean

    Returns
    -------
    out : String
        comment -- String after cleanning
    """

    comment = str.lower(comment)                                                        # set all words to lowercase
    comment = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", " ",comment)    # remove emails
    comment = re.sub(r"@\S+", " ",comment)                                              # revove mentions (tag @)
    comment = re.sub(r"https\S+|http\S+|www\.\S+", " ",comment)                         # remove urls
    comment = re.sub(r"0[567]\d{,8}", " ",comment)                                      # remove phone number
    comment = araby.strip_diacritics(comment)                                           # remove arabic diacritics (harakat)
    comment = araby.normalize_alef(comment)                                             # normalize alef (ا أ إ ئ)
    comment = araby.normalize_teh(comment)                                              # normalize teh (ة -> ه)
    comment = araby.normalize_hamza(comment)                                            # normalize hamza (ء)
    comment = re.sub(r"(a-ZA-Z )\1{2,}", r'\1\1', comment)                              # remove duplicated character and keep 2
    comment = re.sub(r'["#$%&\'()*+,\-./:;<=>@\[\\\]^_`{|}~،]', " ",comment)            # remove punctuation marks
    comment = re.sub(r"\s+", " ",comment)                                               # remove extra spaces
    comment = re.sub(r"\A\s+|\s+\Z", "", comment)                                       # remove space at the begining and the end of words

    return comment

ARABIC_STOPWORDS = stopwords.words("arabic")    # list of arabic stopwords
ENGLISH_STOPWORDS = stopwords.words("english")  # list of english stopwords
FRENCH_STOPWORDS = stopwords.words("french")    # list of french stopwords

# cleaning arabic stopwords to match our cleaned data
for i in range(len(ARABIC_STOPWORDS)):
    ARABIC_STOPWORDS[i] = clean_text(ARABIC_STOPWORDS[i])

def remove_stopwords(comment: str) -> str:
    """
    Remove stopwords from comments
    
    Parameters
    ----------
    comment : String
        comment containing stopwords

    Returns
    -------
    out : String
        comment -- String without stopwords
    """
    words = comment.split()         # split comment into words (tokens)
    new_comment = []                # temp list of new comment words
    for word in words:
        if word in ARABIC_STOPWORDS or word in ENGLISH_STOPWORDS or word in FRENCH_STOPWORDS:
            continue
        new_comment.append(word)    # add non stopword words to temp list
    comment = " ".join(new_comment) # join words without stopwords to form clean comment
    return comment