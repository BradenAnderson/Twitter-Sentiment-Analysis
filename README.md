# Twitter-Sentiment-Analysis

### Project Goal: Classify tweets for instances of hate speech. 

***

### Table of Contents

##### [Handling Emojis](#handling-emojis)

##### [Data Cleaning with SpaCy](#data-cleaning-with-spacy)

##### [Vader Sentiment](#vader-sentiment) 

##### [Exploratory Data Analysis](#exploratory-data-analysis) 

##### [Modeling Part 1](#modeling-part1) 

##### [Exploring fastText](#fastText) 

##### [Modeling with fastText word vectors in Scikit-Learn pipeline](#fastText-sklearn) 

***
***

# Handling Emojis
 

**Background Information**

The Python string data type uses the Unicode standard for representing characters. The goal of the Unicode specification is to create a character set that covers all forms of written natural language (including emojis!) where every character is mapped to a unique number called its “code points”. Therefore, a Unicode (Python) string can be viewed as a sequence of code point values, where each code point value is simply a number between 0 – 0x10FFFF hex (1,114,111 decimal) that represents a single character (or emoji). The keys on a standard U.S. keyboard are all represented with Unicode code point values of 127 or less, therefore provided that every tweet in our dataset is written in English (which is a safe assumption for the dataset used for this project) we can identify characters that must be part of an emoji by simply finding the characters where the associated code point value is 128 or above. 

**00_Emoji_Data_Cleaning.ipynb Notebook description**

**Inputs:**
1)	The original unprocessed twitter data
2)	An csv file that contains Unicode code point values and the associated sentiment for various emojis.

The goal of this notebook is to parse through the unprocessed tweets, identify all emojis in the tweets, and replace each emoji with a short string of words that describes the sentiment or meaning that the particular emoji is used to convey. Although there are some python libraries that have been created to accomplish this task, I decided to implement my own solution from scratch. My first step towards accomplishing this task was to create the file “emoji_partial.csv”, which contains a table that cross references the code point values for over 200 different emojis to a short string of words that I believe capture the sentiment that each individual emoji is used to convey. 
The 00_Emoji_Data_Cleaning.ipynb notebook imports the emoji_partial.csv file into a pandas DataFrame and then builds a dictionary where each key is a sequence of characters that represents an emoji, and the value is the sentiment associated with that emoji. Then, using the method described in the background information, each tweet is parsed and the emojis are replaced by the associated sentiment. Extra care was taken to capture special situations such as multiple emojis existing one after another (causing their character sequences run together) using regular expressions.

If, during the parsing phase, an emoji is encountered that is not one of the ~200 that I manually created a sentiment string for, then the associated emoji code is added to a separate output file “unknown_emoji.csv” and the emoji is simply removed from the tweet. The unknown_emoji.csv file is a helpful list if I ever decide to go back and create sentiment strings for additional emojis. 

Note: There is another file “emoji_full_no_sentiment.csv” that contains the names and Unicode code point values for over 3000 different emojis without any sentiment strings. Feel free to use this file if you ever have a project where you want to replace emojis with a custom set of sentiment strings. 

After all emojis were replaced by their associated sentiment string, this notebook concludes by outputting a new file “train_tweets_with_emojis_clean.csv”, which will be the starting point for the next data cleaning steps. 


***

##### Data Cleaning with SpaCy 
<a name="data-cleaning-with-spacy"></a>

**Background**

The goal of natural language processing is to use machine learning techniques to create models that can understand and derive insight from written human languages. Since mathematical models can only operate on numerical inputs, the first step in training such a model is to find a method of mapping portions of natural language text to numerical values. There are several different methods for how to approach this preprocessing step, many of which involve splitting each input document into word level “tokens”, where the set of all unique tokens across all training documents is referred to as the vocabulary. Using this technique, a features matrix can be constructed where each document in the training set is represented as a row, and the number of columns is equal to the number of words in the vocabulary. In the simplest possible model, each row in the feature’s matrix could simply contain a count of how many times each word in the vocabulary occurred in that particular document. 

This simple model suffers from several drawbacks (e.g. counting word frequencies alone does not account for contextual information such as word ordering. Also, the frequency of a word is often times not a good metric to determine how much it contributes to the overall meaning of the document as common words like "a" and "the" may be very frequent but carry little meaning) but it is enough to illustrate one important point that holds true even for more complex methods. As the size of the vocabulary increases, the dimensionality of the feature’s matrix increases which causes the amount of training data that is required for a model to accurately generalize to increase greatly. Furthermore, as the vocabulary increases the memory and computational resources required to process the training matrix increases. Lastly, as the size of the vocabulary becomes increasing large compared to the average document length (which is relatively short for tweets) the features matrix becomes increasingly sparse, and we may find that for words that occur the most infrequently there is simply not enough training data to accurately learn how the word impacts the desired target. 

**01_Data_Cleaning_With_Spacy Notebook Description**

**Inputs:**

1)	The tweet dataset that been previously processed to replace emojis with their associated sentiment (train_tweets_with_emojis_clean.csv)
2)	A file that contains a table of common contracts and their associated expanded forms (contractions.csv).
3)	A file that contains common slang or “sms speak” that often shows up in conversations on the internet, along with the associated expanded form or meaning (sms_speak.csv).


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

##### Vader Sentiment

**Background**

Two general categories of techniques for solving sentiment analysis problems are **machine learning** (i.e. statistical) and **rules-based** approaches. Depending on which approach is taken, various trade offs will be made. For example, a machine learning style approach requires significant training data and processing time, and even after training is accomplished the models ability to successfully generalize to new data hinges greatly on the quality of the data it was trained on (how representative of the overall population the training data really is). Additionally, many machine learning models are “black box” meaning it is difficult to understand and interpret the learned relationships between the input text and output classification, which can make it difficult to modify or extend a model’s capabilities. In contrast, a rules-based approach does not require training data but rather depends on the development of comprehensive sentiment labeled lexicons and rules that that define how grammatical and contextual relationships impact the overall sentiment. Development of a sentiment labeled vocabulary and useful grammatical rules is both difficult and time consuming, however once it has been developed such a system may allow for faster, more interpretable analysis, and may be easier to extend or modify to fit different applications. 

VADER (Valence Aware Dictionary for sEntiment Reasoning) is considered a gold standard rules-based sentiment analysis tool that was created and validated with a "wisdom of the crowd" approach via workers on the Amazon Mechanical Turk platform. VADERs lexicon contains over 7,500 words with defined sentiment polarity (positive or negative) as well as intensity (-4 to +4) and is specifically designed to be effective on short social media style text (even including sentiments for a full set of emojis and emoticons). In addition to the sentiment lexicon, VADER uses a set of five rules to define how various grammatical and contextual aspects of text modify the overall sentiment, these rules are summarized as follows: 

1.	Exclamation points increase sentiment intensity.
2.	Excess capitalization (e.g. ALL CAPS) increases sentiment intensity.
3.	Degree modifiers or ‘booster’ words alter the sentiment intensity of the word they describe (e.g. ‘extremely’ or ‘marginally’).
4.	Contrastive conjunctions (e.g. ‘but’) signals a shift in sentiment, and the second half of the sentiment is considered dominant. 
5.	Negation terms flip the sentiment intensity of the word they describe (e.g. ‘isn’t great’). 

When the VADER sentiment analyzer is applied to a text four scores are generated. The first three (positive, negative and neutral) always sum to 1 and describe the relative proportions of the sentiment that fall into each of the three categories. The fourth is the compound score which is calculated by adding up the valence scores for each word in the lexicon, adjusting the scores according to the rules above, and them normalizing the result to fall between -1 (most negative sentiment) and 1 (most positive sentiment). The compound score is the best number to use when determining the overall sentiment of a text.

For a full description of VADER please reference the following resources: 

1. [VADER research paper](http://eegilbert.org/papers/icwsm14.vader.hutto.pdf) 
2. https://github.com/cjhutto/vaderSentiment

   
**02_vaderSentiment and 02_2_vaderSentiment notebook description**

**Inputs:** 
1) The tweet data file generated by the 01_Data_Cleaning_With_Spacy notebook (train_tweets_spacy_clean.csv)

In the vaderSentiment notebooks I explored using VADER as a tool for analyzing tweets to determine whether or not the text they contain constitutes hate speech. To accomplish this, I calculated the compound sentiment score for each tweet and then specified a classification threshold, with compound scores below this threshold receiving the hate speech classification while compound scores above the threshold were classified as not hate speech. I then used various classification metrics (sensitivity, precision, specificity, accuracy and F1-score) as a means of quantifying how successful VADER is at finding instances of hate speech. Due to the heavy class imbalance in the tweet dataset (with non-hate speech examples greatly out number hate speech) the F1-score was the primary metric of interest. 

Clearly, in order to obtain the best possible classification results it is important to carefully select the threshold value. It is also worth considering how various text preprocessing steps could impact VADERs ability to successfully perform classifications. With the goal of benchmarking VADERs capabilities as a hate speech classifier, 3200 classification simulations were performed by varying the following settings:

1.	Tweets preprocessed to remove twitter handles (True or False).
2.	Tweets preprocessed to remove websites (True or False).
3.	Whether a single compound score was calculated for the entire tweet, or the tweet was first broken up into sentence level tokens, each of which receiving a compound score, and then averaging the sentence level compound scores to make the classification decision.
4.	400 different threshold levels ranging from -0.99 (don’t classify anything as hate speech) to 0.99 (classify everything as hate speech).

Several plots were created along with significant analysis that explores and interprets the test results (see the 02_2_vaderSentiment notebook) which I can not fully recreate here. However, at a high level the results showed that the only preprocessing step that impacted the final classification result was the sentence level analysis decision. I also showed that for full tweet analysis the F1-score had a maximum value of 0.226 which was achieved when the decision threshold was set to approximately -0.295. For sentence level analysis the maximum F1-score was 0.221, which was achieved using a decision threshold of -0.131. 

Since a VADER compound score of 0 indicates complete neutrality, it does make sense that the ideal classification threshold for finding hate speech should be a negative value. Intuitively, we may have anticipated the ideal decision threshold to be even more negative, as I think most people would agree statements that are truly hateful (i.e. sexist or racist) are more than just slightly negative. Furthermore, the low F1-scores indicate that the VADER compound is not performing effectively as a hate speech classifier.

One key to understanding why VADER was ineffective at classifying hate speech is found by viewing the distribution of compound scores for hate speech vs non hate speech tweets. We observed that regardless of class, the vast majority of tweets received neutral compound scores, which indicates that the classes cannot be effectively separated with this type of simple decision boundary.

**Outputs:** 
1.	vaderSentiment_Analysis.csv – A file containing classification metrics resulting from using VADER to classify tweets with various decision thresholds and data cleaning decisions.
2.	vader_no_data_cleaning.csv, vader_handles_removed.csv, vader_websites_removed.csv, vader_sentence_level.csv, vader_full_preprocessing.csv – All of these files contain the tweet dataset along with the VADER sentiment scores when the associated data cleaning steps were applied.
3.	vader_full_preprocessing_model.csv. This is the same data as vader_full_preprocessing.csv except the VADER compound scores have been shifted to take on values of 0 to 2 rather than the standard -1 to 1. This was done to facilitate using the compound score as an input feature to a Naïve Bayes model, as Naïve Bayes requires that all inputs are positives numbers. 


***

##### [Exploratory Data Analysis](#exploratory-data-analysis) 

***

##### [Modeling Part 1](#modeling-part1) 

***

##### [Exploring fastText](#fastText)

***

##### [Modeling with fastText word vectors in Scikit-Learn pipeline](#fastText-sklearn) 

***
