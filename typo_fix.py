import string
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words
# from pattern.en import suggest
from autocorrect import Speller
from collections import Counter

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.drop(['beer/beerId', 'user/ageInSeconds', 'user/birthdayRaw', 'user/birthdayUnix', 'user/gender', 'user/profileName', 'review/timeStruct', 'review/timeUnix'], axis=1, inplace=True)
test_df.drop(['beer/beerId', 'user/ageInSeconds', 'user/birthdayRaw', 'user/birthdayUnix', 'user/gender', 'user/profileName', 'review/timeStruct', 'review/timeUnix'], axis=1, inplace=True)

train_df.dropna(subset=['review/text'], inplace=True)

spell_correct = Speller(lang='en')
stop = set(stopwords.words('english'))
porter = PorterStemmer()

spelling_dict = dict()

def text_process(review):
    def is_number(word):
        return word.replace('.','',1).isdigit()
    
    def convert_number(number):
        if is_number(number):
            return num2words(number)
        else:
            return number
    
    def reduce_lengthening(text):
        # replace aaa... by aa
        # spellllling -> spelling
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)
    
    def spelling_correct(word):
        global spelling_dict
        word = reduce_lengthening(word)
        if spelling_dict.get(word) is None:
            correction = spell_correct(word)
            spelling_dict[word] = correction
            return correction
        else:
            return spelling_dict[word]
    
    # remove all punctuations except apostrophe
    # apostrophe is omitted because don't-> dont won't be identified as stopword otherwise
    punctuations = ''.join([p for p in string.punctuation if p != '\''])
    table = str.maketrans('', '', punctuations)
    review = str(review).translate(table)
    
    # convert text to lower case
    review = str(review).lower()
    
    # remove stop words
    review = ' '.join(list(filter(lambda word : word not in stop, review.split())))
    
    # remove apostrophe
    table = str.maketrans('', '', '\'')
    review = review.translate(table)
    
    # separate numbers from units: for example 8oz to 8 oz
    review = ' '.join(re.split('(\d+)', review))
    
    # replace numbers to their spellings. for example 100 -> one hundred
    review = ' '.join([convert_number(word) for word in review.split()])
    
    return review

train_df['processed_text'] = train_df['review/text'].map(text_process)


review_text = ' '.join(train_df['processed_text'].tolist())

word_count = Counter()
word_count.update((word for word in review_text.split()))

correct_dict = dict()

for word, count in word_count.most_common():
    # print(f"{word}, {count}", end=" ")
    word = word.lower()
    corr = spell_correct(word.lower())
    if corr != word:
        answer = input(f"was: {word}, correct: {corr}")
        if answer == "":
        	continue
        elif answer == 'y':
        	correct_dict[word] = corr
        elif answer == '**':
        	print(correct_dict)
        else:
        	correct_dict[word] = answer

print(correct_dict)