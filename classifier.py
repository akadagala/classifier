from nltk.corpus import twitter_samples, wordnet, stopwords
stop_words = stopwords.words("english")
import re, nltk, string, random
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
lemmatizer = WordNetLemmatizer()
from nltk import FreqDist, classify, NaiveBayesClassifier

import datasetone

"""
Takes a string and tokenizes it (splits into tokens 
    on whitespace)
"""
def tokenize(string):
    return string.split(" ")

"""
Removes special characters from one token
"""
def remove_spec_char(token):
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
    token = re.sub("(@[A-Za-z0-9_]+)","", token)
    return token

"""
Get the wordnet POS tag for a given token, in format that 
    lemmatizer accepts, helper function for lemmatize_sentence
"""
def get_wordnet_pos(word):
    #Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

"""
Given a token (word), return lemmatized version
"""
def lemmatize_word(token):
    print("running lemmatizer...")
    return lemmatizer.lemmatize(token, get_wordnet_pos(token))

"""
Normalization process for one set of tokens (sentence) in 
    text, includes removing special characters, lemmatizing,
    and filtering stopwords/punctuation
"""
def normalize(tokens):
    normalized_tokens = []
    words_w_tags = pos_tag(tokens)

    for token, tag in words_w_tags:
        #print("removing special characters...")
        token = remove_spec_char(token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        #print("lemmatizing token...")
        token = lemmatizer.lemmatize(token, pos)

        #add to list if token is not punctuation or stop word
        if token not in string.punctuation and token not in stop_words:
            normalized_tokens.append(token.lower())
    
    return normalized_tokens

"""
Given a list of cleaned tokens lists, return those tokens tagged w/ True,
    for the purpose of feeding it into a model
"""
def create_tagged_word_list(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

"""
Use data sets to train a Naive Bayes Classifier
"""
def build_classifyer():
    # the first source of data used in this program is a collection of tweets provided
    # by the nltk library 
    pos_tweets_tokenized = twitter_samples.tokenized('positive_tweets.json')
    neg_tweets_tokenized = twitter_samples.tokenized('negative_tweets.json')

    # the second source is a collection of tagged text messages sourced from kaggle.com
    # this data set is multiple thousands lines long
    datasetone_pos = datasetone.get_positive_sentences()
    datasetone_neg = datasetone.get_negative_sentences()

    pos_normalized = []
    neg_normalized = []
    
    # add the tweets to their respective lists
    for tweet_tokens_list in pos_tweets_tokenized:
        pos_normalized.append(normalize(tweet_tokens_list))

    for tweet_tokens_list in neg_tweets_tokenized:
        neg_normalized.append(normalize(tweet_tokens_list))

    # do the same with the messages
    for sentence in datasetone_pos:
        pos_normalized.append(normalize(tokenize(sentence)))
    
    for sentence in datasetone_neg:
        neg_normalized.append(normalize(tokenize(sentence)))

    # create tagged lists that is usable for model training 
    positive_tagged = create_tagged_word_list(pos_normalized)
    negative_tagged = create_tagged_word_list(neg_normalized)

    #gather all positive words into one set 
    positive_words = []
    for sentence in pos_normalized:
        for token in sentence:
            positive_words.append(token)

    # uncomment the line below to create a frequency distribution for positive words, 
    # for visual analysis purposes
    #freq_dist_pos = FreqDist(positive_words)

    # tag each sentence with their respective sentiment
    positive_dataset = [(sent_dict, "Positive")
                     for sent_dict in positive_tagged]

    negative_dataset = [(sent_dict, "Negative")
                        for sent_dict in negative_tagged]

    # create a data set with everything combined and shuffle it
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    # split the data set into testing and training sets
    index = len(dataset) // 2
    train_data = dataset[:index]

    # uncomment the two lines below to test the accuracy of the model
    #test_data = dataset[index:]
    #print("Accuracy is:", classify.accuracy(classifier, test_data))

    return NaiveBayesClassifier.train(train_data)

"""
Given a sentence, use a Naive-Bayes classifier to classify a sentence as 
    negative or positive
"""
def classify_sentence(sentence):
    classifier = build_classifyer()
    tokens = tokenize(sentence)
    return classifier.classify(dict([token, True] for token in tokens))