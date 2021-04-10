# Twitter-Sentiment-Analysis

### Project Goal: Classify tweets for instances of hate speech. 

***

### Table of Contents

##### [Handling Emojis](#emoji-cleaning)

##### [Data Cleaning with SpaCy](#text-data-cleaning)

##### [Vader Sentiment](#vaderSentiment) 

##### [Exploratory Data Analysis](#exploratory-data-analysis) 

##### [Modeling Part 1](#modeling-part1) 

##### [Exploring fastText](#fastText) 

##### [Modeling with fastText word vectors in Scikit-Learn pipeline](#fastText-sklearn) 

***
***

##### [Handling Emojis](#emoji-cleaning)

Inputs:
1)	The original unprocessed twitter data
2)	An csv file that contains Unicode code point values and the associated sentiment for various emojis.

**Background Information**

The Python string data type uses the Unicode standard for representing characters. The goal of the Unicode specification is to create a character set that covers all forms of written natural language (including emojis!) where every character is mapped to a unique number called its “code points”. Therefore, a Unicode (Python) string can be viewed as a sequence of code point values, where each code point value is simply a number between 0 – 0x10FFFF hex (1,114,111 decimal) that represents a single character (or emoji). The keys on a standard U.S. keyboard are all represented with Unicode code point values of 127 or less, therefore provided that every tweet in our dataset is written in English (which is a safe assumption for the dataset used for this project) we can identify characters that must be part of an emoji by simply finding the characters where the associated code point value is 128 or above. 

**00_Emoji_Data_Cleaning.ipynb Notebook description**

The goal of this notebook is to parse through the unprocessed tweets, identify all emojis in the tweets, and replace each emoji with a short string of words that describes the sentiment or meaning that the particular emoji is used to convey. Although there are some python libraries that have been created to accomplish this task, I decided to implement my own solution from scratch. My first step towards accomplishing this task was to create the file “emoji_partial.csv”, which contains a table that cross references the code point values for over 200 different emojis to a short string of words that I believe capture the sentiment that each individual emoji is used to convey. 
The 00_Emoji_Data_Cleaning.ipynb notebook imports the emoji_partial.csv file into a pandas DataFrame and then builds a dictionary where each key is a sequence of characters that represents an emoji, and the value is the sentiment associated with that emoji. Then, using the method described in the background information, each tweet is parsed and the emojis are replaced by the associated sentiment. Extra care was taken to capture special situations such as multiple emojis existing one after another (causing their character sequences run together) using regular expressions.

If, during the parsing phase, an emoji is encountered that is not one of the ~200 that I manually created a sentiment string for, then the associated emoji code is added to a separate output file “unknown_emoji.csv” and the emoji is simply removed from the tweet. The unknown_emoji.csv file is a helpful list if I ever decide to go back and create sentiment strings for additional emojis. 

Note: There is another file “emoji_full_no_sentiment.csv” that contains the names and Unicode code point values for over 3000 different emojis without any sentiment strings. Feel free to use this file if you ever have a project where you want to replace emojis with a custom set of sentiment strings. 

After all emojis were replaced by their associated sentiment string, this notebook concludes by outputting a new file “train_tweets_with_emojis_clean.csv”, which will be the starting point for the next data cleaning steps. 


***

##### [Data Cleaning with SpaCy](#text-data-cleaning)

Inputs:

1)	The tweet dataset that been previously processed to replace emojis with their associated sentiment (train_tweets_with_emojis_clean.csv)
2)	A file that contains a table of common contracts and their associated expanded forms (contractions.csv).
3)	A file that contains common slang or “sms speak” that often shows up in conversations on the internet, along with the associated expanded form or meaning (sms_speak.csv).

**Background**

The goal of natural language processing is to use machine learning techniques to create models that can understand and derive insight from written human languages. Since mathematical models can only operate on numerical inputs, the first step in training such a model is to find a method of mapping portions of natural language text to numerical values. There are several different methods for how to approach this preprocessing step, many of which involve splitting each input document into word level “tokens”, where the set of all unique tokens across all training documents is referred to as the vocabulary. Using this technique, a features matrix can be constructed where each document in the training set is represented as a row, and the number of columns is equal to the number of words in the vocabulary. In the simplest possible model, each row in the feature’s matrix could simply contain a count of how many times each word in the vocabulary occurred in that particular document. 

This simple model suffers from several drawbacks (e.g. counting word frequencies alone does not account for contextual information such as word ordering. Also, the frequency of a word is often times not a good metric to determine how much it contributes to the overall meaning of the document as common words like "a" and "the" may be very frequent but carry little meaning) but it is enough to illustrate one important point that holds true even for more complex methods. As the size of the vocabulary increases, the dimensionality of the feature’s matrix increases which causes the amount of training data that is required for a model to accurately generalize to increase greatly. Furthermore, as the vocabulary increases the memory and computational resources required to process the training matrix increases. Lastly, as the size of the vocabulary becomes increasing large compared to the average document length (which is relatively short for tweets) the features matrix becomes increasingly sparse, and we may find that for words that occur the most infrequently there is simply not enough training data to accurately learn how the word impacts the desired target. 

**01_Data_Cleaning_With_Spacy Notebook Description**

The goal of this project is to build a model that can understand the sentiment of a tweet and determine if it contains hate speech. The first step in building such a model is to establish the vocabulary of the training data, and as described in the background information, it is desireable to establish a vocabulary that is as small as possible while retaining the underlying sentiment. 

As such, the focus of this data cleaning notebook is the creation of this minimum, sentiment dense, vocabulary followed by transforming the raw tweet text data to make it conform to this standardized system. The process for accomplishing this is as follows:

1) Load the pretrained spaCy model en_core_web_md, which will be used for its tokenization and lemmatization capabilities. 
2) Preprocess the text data
    - Convert all contractions found in contractions.csv to their expanded form.
    - Convert all slang "sms speak" terms found in sms_speak.csv to their expanded/proper forms. 
    - Remove punctuation marks (preserve # symbols since we are working with twitter data).
3) Use spaCys tokenizer to break each tweet into word level tokens.
4) Use spaCys lemmatizer to convert each word token to its lemma. 

   **Note:** Lemmatization is the process of grouping together inflected or derivationally related words to a common root form. 
   For example, a lemmatizer may perform the following word mappings:
       am, are, is --> be
       walk, walks, walked, walking --> walk
       
6) Discard tokens that are too generic to provide sentiment information (e.g. stop words and generic @User twitter handles). 


This notebook concludes by outputting a new file, train_tweets_spacy_clean.csv, that contains the result of applying the data cleaning steps described above to the train_tweets_with_emojis_clean.csv input file.


***

##### [Vader Sentiment](#vaderSentiment) 

***

##### [Exploratory Data Analysis](#exploratory-data-analysis) 

***

##### [Modeling Part 1](#modeling-part1) 

***

##### [Exploring fastText](#fastText)

***

##### [Modeling with fastText word vectors in Scikit-Learn pipeline](#fastText-sklearn) 

***
