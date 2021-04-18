# Twitter-Sentiment-Analysis

### Project Goal: Classify tweets for instances of hate speech. 

***

### Table of Contents

[Handling Emojis](#handling-emojis)

[Data Cleaning with SpaCy](#data-cleaning-with-spacy)

[Vader Sentiment](#vader-sentiment) 

[Exploratory Data Analysis](#exploratory-data-analysis) 

[Modeling Part 1](#modeling-part-1) 

[Exploring fastText](#exploring-fastText) 

[Modeling with fastText word vectors in Scikit-Learn](#modeling-with-fastText-word-vectors-in-scikit-learn) 


***
***

### Handling Emojis
 

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

#### Data Cleaning with SpaCy 

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

### Vader Sentiment

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

### [Exploratory Data Analysis](#exploratory-data-analysis) 

When it comes to exploratory data analysis, a picture really is worth a thousand words, so please, go check it out! For those who decide not to dive into the notebook, a brief summary of the data exploration methods I used is as follows: 

First, I created two separate word clouds – one using the words found in hate speech tweets, and the other using the words found in non-hate speech tweets. Words like ‘happy’, ‘love’, ‘significant’ and ‘friend’ pop out as commonly used terms in non-hate speech tweets, as for the hate speech terms... I’ll leave that for the people who open the notebook. 

Next, I create a slightly different view on this same information by using bar graphs to display the frequency of the top 15 terms used in each class of tweet. I also create a slightly different spin on this same plot by isolating hashtags and again plotting the frequencies of the top 15. 

Finally, in what I thought was the most insightful part of this exploration, I revisit our friend VADER and create 6 more bar graphs and 6 more word clouds. These plots displayed words from the following six categories:

1.	Tweets VADER gave a positive sentiment score to, and were not hate speech
2.	Tweets VADER gave a neutral sentiment score to, and were not hate speech
3.	Tweets VADER gave a negative sentiment score to, and were not hate speech
4.	Tweets VADER gave a positive sentiment score to, and were hate speech
5.	Tweets VADER gave a neutral sentiment score to, and were hate speech
6.	Tweets VADER gave a negative sentiment score to, and were hate speech

In my opinion, the most interesting visualizations are the ones that showing words used in hate speech, yet the tweet received a positive or neutral sentiment from VADER. Inspection of these charts reveals a clear trend, specifically, that some politically and racially related terms that commonly appear in hate speech are not properly accounted for in the VADER lexicon. Armed with this knowledge, I believe we could make some custom additions to VADERs vocabulary that would significantly increase its utility as a hate speech classifier.


***

### [Modeling Part 1](#modeling-part-1) 

For this round of modeling, all inputs (features matrices) were sparse document vectors (length of each vector is the vocabulary length) created by Scikit-Learns TfidfVectorizer. Using these TFIDF weighted vectors as inputs I created and tuned several different model types to compare their relative performance. To address the significant class imbalance (where the training data has significantly more examples of non-hate speech than hate speech) I explored using the imblearn library to hyper-parameterize the use of various over sampling techniques (e.g. random oversampling, synthetic minority oversampling SMOTE, adaptive synthetic sampling ADASYN) for several of the models.

I also explored using the VADER sentiment scores as additional input features to the supervised models, however I was quickly able to show that doing this did not lead to an increased performance and therefore I did not carrying the idea forward. As shown in previous sections, there is a significant portion of the hate speech terminology that is missing from VADERs vocabulary, therefore it is not surprising that the “out of the box” VADER scores lack predictive power for this task. If the VADER vocabulary was updated to include some task specific terms, it may be interesting to once again explore using the VADER scores as inputs to supervised learning models. 

To maintain a sense of organization and to easily spread out the modeling work load, a separate jupyter notebook was utilized for creating and tuning each unique model type. All notebooks from this round of modeling begin with the “04” naming convention, a full list of which is as follows: 

04_Multinomial_Naive_Bayes_Modeling.ipynb
04_SGD_Modeling.ipynb
04_Complement_Naive_Bayes_Modeling.ipynb
04_Extra_Random_Forest.ipynb
04_Gradient_Boosted_Forest_Models.ipynb  
04_Logistic_Regression.ipynb
04_MLP.ipynb

Note: Most of the trained models (pickle files) for this round of modeling are available in the folder "04_trained_model_pickle_files". The few that are missing were too large to upload to GitHub. 

To easily compare relevant classification metrics across model types, another notebook **04_Modeling_Analysis.ipynb** was created with the sole purpose of importing and displaying the GridSearch results created in the notebooks above. 

After several rounds of model tuning, I isolated a group of five models that had the highest performance as determined by the GridSearch cross validation F1 score. These models are as follows: 

1.	Naïve Bayes with minority class random over sampler (naïve_bayes_ros_gs2.pkl), cross validation f1 score = 0.73
2.	Support vector machine with minority class random over sampler (SGD_ros_nv_gs1.pkl), cross validation f1 = 0.72
3.	Gradient Boosted Random Forest with 1250 decision trees and minority class random over sampler (bf_ros_gs1.pkl), cross validation f1 score = 0.679
4.	Logistic Regression with minority class random over sampler (lr_ros_gs1.pkl), cross validation f1 score = 0.73
5.	Multilayer Perceptron with 100 hidden units (file too large to include), cross validation f1 score = 0.73 

Note: The full list of hyper parameters used in each model are not reproduced here. To see this information please reference the modeling and modeling analysis notebooks. 

Once I had identified my top performing models as determined by cross validation, I then tested each model using a completely unseen dataset of 17198 tweets. The results are as follows:

1.	Naïve Bayes, F1 score = 0.768
2.	Support vector machine, F1 score = 0.723
3.	Gradient boosted random forest, F1 score = 0.699
4.	Logistic Regression, F1 score = 0.765
5.	Multilayer Perceptron, F1 score = 0.769

Next, I wrote a python script to automate creating ensembles with all possible combinations of the above models. The ensembles were all based on the scikit-learn voting classifier with a hard voting scheme. To cover all possible combinations, the ensemble size was ranged from 2 to 5, for a total of 26 possible ensemble models. The code that automates instanting, fitting, and outputting test results for all 26 ensembles can be found in the notebook 04_Generate_Hard_Voting_Ensemble_Predictions.ipynb. 

Note: Hard voting is where each classifier in the ensemble gets a single vote, and the final predicted class is a simple majority rule of the predicted classes for each ensemble member. The alternative option is soft voting, where each classifier outputs a set of probabilities for target class, and the final prediction made by the ensemble is the class with the highest overall probability. I’ll also note that, in general, for a set of well-tuned classifiers soft voting should be preferred as it lets classifiers that are more ‘certain’ on a given datapoint weigh in more heavily. However, not all model types have the ability to generate probabilistic outputs (e.g. a support vector machine with the perceptron loss function does not), therefore to be inclusive for all five model types hard voting was chosen. 

The F1 scores calculated using the set of 17198 previously unseen tweets are shown below.

**Size two hard voting ensemble F1 Scores:** 

1.	Naïve Bayes and Logistic Regression, F1 Score = 0.768
2.	Naïve Bayes and Support Vector Machine, F1 Score = 0.756
3.	Naïve Bayes and Gradient Boosted Random Forest, F1 Score = 0.755
4.	Naïve Bayes and Multilayer Perceptron, F1 Score = 0.772
5.	Logistic Regression and Support Vector Machine, F1 Score = 0.754
6.	Logistic Regression and Gradient Boosted Random Forest, F1 Score = 0.753
7.	Logistic Regression and Multilayer Perceptron, F1 Score = 0.771
8.	Support Vector Machine and Gradient Boosted Random Forest, F1 Score = 0.720
9.	Support Vector Machine and Multilayer Perceptron, F1 Score = 0.772
10.	Gradient Boosted Random Forest and Multilayer Perceptron = 0.745

**Size three hard voting ensemble F1 Scores:**

1.	Naïve Bayes, Logistic Regression, Support Vector Machine, F1 Score = 0.773
2.	Naïve Bayes, Logistic Regression, Gradient Boosted Random Forest, F1 Score = 0.7878
3.	Naïve Bayes, Logistic Regression, Multilayer Perceptron, F1 Score = 0.779
4.	Naïve Bayes, Support Vector Machine, Gradient Boosted Random Forest, F1 Score = 0.772
5.	Naïve Bayes, Support Vector Machine, Multilayer Perceptron, F1 Score = 0.780
6.	Naïve Bayes, Multilayer Perceptron, Gradient Boosted Random Forest, F1 Score = 0.784
7.	Logistic Regression, Support Vector Machine, Multilayer Perceptron, F1 Score = 
8.	Logistic Regression, Support Vector Machine, Gradient Boosted Random Forest, F1 Score = 
9.	Logistic Regression, Multilayer Perceptron, Gradient Boosted Random Forest, F1 Score =
10.	Support Vector Machine, Multilayer Perceptron, Gradient Boosted Random Forest, F1 Score =

**Size four hard voting ensemble F1 Scores:**

1.	Naïve Bayes, Support Vector Machine, Logistic Regression, Gradient Boosted Random Forest, F1 Score = 0.772
2.	Naïve Bayes, Logistic Regression, Support Vector Machine, Multilayer Perceptron, F1 Score = 0.7675
3.	Naïve Bayes, Logistic Regression, Gradient Boosted Random Forest, Multilayer Perceptron, F1 Score = 0.773
4.	Naïve Bayes, Support Vector Machine, Gradient Boosted Random Forest, Multilayer Perceptron, F1 Score = 0.772
5.	Logistic Regression, Support Vector Machine, Gradient Boosted Random Forrest, Multilayer Perceptron, F1 Score = 0.758

**Size five hard voting ensemble F1 Score:**

1.	Naïve Bayes, Support Vector Machine, Gradient Boosted Random Forest, Logistic Regression and Multilayer Perceptron, F1 Score = 0.7588

Next, to explore the effect of probabilistic ensemble voting, the support vector machine model was removed and all possible ensemble combinations of the remaining four models were created, this time using the soft voting scheme. This process was automated similar to the hard voting ensembles, see the notebook "04_Generate_Soft_Voting_Ensemble_Predictions" for the code that does this. With four models to choose from the possible ensemble sizes are 2, 3, or 4, which leads to a total of 11 possible ensembles. 

**Size two soft voting ensemble F1 Scores:**

1.	Naïve Bayes and Logistic Regression, F1 Score = 0.7807
2.	Naïve Bayes and Gradient Boosted Random Forest, F1 Score = 0.779
3.	Naïve Bayes and Multilayer Perceptron, F1 Score = 0.776
4.	Logistic Regression and Gradient Boosted Random Forest, F1 Score = 0.763
5.	Logistic Regression and Multilayer Perceptron, F1 Score = 0.775
6.	Gradient Boosted Random Forest and Multilayer Perceptron, F1 Score = 0.772

**Size three soft voting ensemble F1 Scores:**

1.	Naïve Bayes, Logistic Regression, Gradient Boosted Random Forest, F1 Score = 0.7934
2.	Naïve Bayes, Logistic Regression, Multilayer Perceptron, F1 Score = 0.7732
3.	Naïve Bayes, Gradient Boosted Random Forest, Multilayer Perceptron, F1 Score = 0.7914
4.	Logistic Regression, Multilayer Perceptron, Gradient Boosted Random Forest, F1 Score = 0.770

**Size four soft voting ensemble F1 Score:**

1.	Naïve Bayes, Logistic Regression, Gradient Boosted Random Forest, Multilayer perceptron, F1 Score = 0.777

In summary, the best F1 score was achieved by soft vote ensembling three models, which were Naive Bayes, Logistic Regression, and Gradient Boosted Random Forest. This ensemble achieved an F1 score of 0.7934, which at the time of this writing ranks #30 (out of 16171 registered particpants and a leader board that holds 1186 top scores) on the Analytics Vidhya Twitter Sentiment Analysis competition leader board. 

At this point we have created, tuned, and ensembled various model types, all of which used the Scikit-Learn TFIDF vectorizer as the final step in preparing our twitter data for the machine learning model. I believe there are a few potential methods that could increase our predictive ability even further, these are: 

1. Revisit the data cleaning decisions to see what improvements can be made. Implementing a spellchecker that is efficient and accurate on social media style text is one possible method of improvement. Another could be revisiting VADERs vocabulary to try leveraging its capabilities again.

2. Try different word embedding techniques. Pretrained word vectors could potentially be useful (e.g. GloVe), or we could consider using a method of training our own application specific word vectors (e.g. fastText or word2vec). 

3. Try a deep learning approach. Reccurrent neural networks can be very effective at natural language processing tasks, and these networks have the ability to utilize pretrained word embeddings (e.g. fastText vectors) or to learn their own embeddings via a network embedding layer.

Several of these options will be explored in the work that follows, so if that sounds interesting, please stick around and check it out! 

***

### [Exploring fastText](#exploring-fastText)

**Background:** fastText is an open-source library created by the Facebook AI research team that has the capability to be used both as a complete text classification model, and as a means of generating word vector representations which can then be used as inputs to other supervised machine learning models. When used as a supervised learning classifier, fastText offers easy interfaces for common tasks such as hyperparameter tuning, generating predictions on new data, and saving/loading trained models. When used in an unsupervised setting to learn word vector representations there are two model types available, skipgram and cbow (continuous-bag-of-words). These models differ in the way that the particular context a word is used in impacts the vector that is ultimately generated for that word.
For more information on fastText, please reference the following resources: 

[fastText API](https://fasttext.cc/docs/en/html/index.html) 

[fastText docs](https://fasttext.cc/docs/en/support.html ) 

**05_fastText notebook description:**

Up to this point in the project, I have been creating models to classify tweets for the presence of hate speech by representing each tweet in my training set as an inverse document frequency (idf) weighted count vector using the Scikit-Learn TfidfVectorizer. After creating and tuning several types of models using this scheme, I then set out to find alternate methods of creating vector representations of my input tweets, which led me to fastText.  Since I had never used this tool before, the 05_fastText notebook documents my initial exploration into all things fastText. 

I began by exploring fastTexts supervised learning capabilities. It seems that part of what lets fastText live up to its name is the strict requirements it places on how the text data and labels need to be formatted before using them as inputs for training. After carefully constructing my data into the required format (example: `__label__0text`), I used scikit learns train test split to create a training set and a stratified validation set and saved each as a .txt file (reference files: train_data.txt and test_data.txt). 

Next, I explored letting fastText use the validation file to auto tune the models hyperparameters for varying lengths of time. After a 10-minute tuning period the model had a validation set F1 score of about 0.18 (model file: autoTuned_model.bin). Increasing the tuning period to 8.33 hours significantly increased the performance on the validation set, with an F1 score of 0.82 (model file: autoTuned_longTrain_model.bin). Increasing the tuning time to 23 hours further increased the validation set F1 score to a value of 0.85 (model file: 23hr_model.bin). 

I then took both the 8.33 hour and 23-hour models and tested them on truly unseen data, using the approximately 17k tweets in the final testing set. These models received F1-scores of approximately 0.59 and 0.60 respectively, indicating that both the 8.33 hour and 23-hour models had likely become overfit on the validation file. As a point of reference, the best testing set F1-score that any model in the entire competition has received at the time of this writing is 0.87. 

If you are familiar with the Scikit-Learn GridSearch behavior, it may not be immediately obvious what my steps to this point have overlooked. Before a Scikit-Learn GridSearch completes, the model is automatically retrained on all of the data using the best hyperparameters found. fastText does not share this functionality out of the box. When setting up the fastText auto tune, recall that I manually separated out the validation set. My final model is automatically trained using the best hyperparameters found, however the validation set is not incorporated into this final training. To make matters even trickier, fastText does not provide an interface to easily return the optimal parameters found by the auto tune process. This means the solution here is not as simple as accessing a model attribute that contains the optimal parameters found during auto tune and using them to retrain on the entire data set. There are work arounds to extract the parameters found by fastText however they are relatively complex and not part of the core fastText API functionalities, so the details of doing this is currently outside the scope of this document. 

Note: An example of a work around that extracts the best parameters found by auto tune from the trained fastText model is shown in the notebook 05_fastText_parameters_from_auto_tune. In that notebook I extract the best parameters from the model that was trained for 8.33 hours and retrain using the entire dataset. I then format the predictions and submit them to Analytics Vidhya for grading. The measured unseen data F1 score for the model trained on all of the data was similar to the score for the model that was trained on all of the data minus the validation set. Again, the fact that the model was able to achieve such a significantly higher F1 score on the validation set as compared with new data indicates that the model had become overfit to the validation set.

One way to fight this over fitting and improve the performance of these fastText classifiers would be to use a cross-validation metric when auto tuning the hyper parameters rather than simply seeking to maximize performance on a single validation set. Cross validated hyper parameter tuning is also not something that fastText currently offers out of the box, but I believe with enough effort a separate class could be written to accomplish this, which may be an interesting side project for another day. 

Next, I began exploring the use of fastTexts unsupervised learning models to create word vectors. Again, I turned my dataset of tweets into a .txt file with one tweet per line, however this time without using any class labels (file: unlabeled_data.txt). Using this file as an input to the unsupervised skipgram model fastText was able to (quickly) learn word vector representations for every word in my dataset. Using the trained word vector model, you can then do interesting thing such as returning the vectors that are most similar to a given word, even for words that were unseen during training. 

Finally, with some help and inspiration from the book “fastText Quick Start Guide” by Joydeep Bhattacharjee I created a python class that can take each input tweet, and using a reference dictionary of word vectors, will find the average of the word vectors associated with a given tweet. The class then returns this single “tweet vector”, which can be used as the input to any supervised classification model (Note: models that cannot accept negative valued inputs, for example Naïve Bayes, would be the one exception to this as the vectors generated by fastText do have negative valued inputs).  Placing an instance of this class in a Scikit-Learn Pipeline allows us to easily GridSearch and tune hyperparameters for models that use these mean embedded fastText vectors as inputs, just as we did with the models in the previous section. 

I concluded this initial fastText exploration by comparing the word vectors created by the skipgram and cbow model types. To do this I used my mean embedding vectorizer class to feed each set of vectors into four standard supervised learning models (support vector machine, extra random forest, multilayer perceptron and gradient boosted random forest). I found that each pair of supervised models had remarkably similar F1 scores regardless of which type of fastText word vector was used as the input. 


***

### [Modeling with fastText word vectors in Scikit-Learn](#modeling-with-fastText-word-vectors-in-scikit-learn) 

***
