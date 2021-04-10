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
2)	 An excel file that contains Unicode code point values and the associated sentiment for various emojis.

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
