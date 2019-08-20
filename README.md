# Subreddit Classification via NLP

## Executive Summary

#### 1. Problem Statement

Language is an ever evolving mode of communication that we use to convey our thoughts and feelings.  As computer technology and data science continue to evolve,  the importance of Natural Language Processing and its relationship to the endless volumes of text contained throughout the internet will become more expand.

In this project, using current webscraping techniques to extract text from two subreddits, my goal was to create a classification model that would be able to accurately predict, from the provided text, which subreddit the text came from.  For this task, I decided upon two subreddit topics that would be difficult to discern, r/gameofthrones (a subreddit for Game of Thrones, the hit HBO TV series) and r/asoiaf (a subreddit for A Song of Ice and Fire, a book series that inspired the show).  

Using a simple classification model, our goal was to maximize our accuracy score for our predictions.  

#### 2. Description of data

The texts were collected over a 7 day period, from July 5th, 2019 to July 11th, 2019.

SOURCE: 

The data collected and utilized for this project was all obtained via reddit.com under the following subreddit:

[r/asoiaf](https://www.reddit.com/r/asoiaf/)
[r/gameofthrones](https://www.reddit.com/r/gameofthrones/)


VARIABLE DESCRIPTIONS: 

| Feature     | Type   | Description                                                                                                                                                                                 |
|-------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| post_text   | object | Raw text scraped from individual reddit posts that contain all elements of the scraped text, including random symbols, characters, and punctuation.                                         |
| subreddit   | object | The subreddit thread origin of the provided text.                                                                                                                                           |
| parsed_post | object | The same text from the post_text.  Embedded within the text, unseen in the csv and Pandas DataFrame, are scaled values provided by spaCy that are used in the predictive modeling process.  |


The text provided in the posts are user generated and often contain spelling errors, topic-specific nomenclature, and other text not found in a dictionary (e.g. text-emojis).  In many cases, posts only contained a picture or video, which were filtered out of our data, as we were only interested in evaluating subreddits via post text.

Additionally, much of the content has overlapping subject matter (i.e. characters, places, etc.) that will certainly diminish the effectiveness of some of our supposed features.  An unforeseen realization with this data, after data collection had begun, is that r/asoiaf also allows show based discussion, meaning some of the provided texts could just as easily be entered into r/gameofthrones.


SIZE: 

1542 observations over 3 variables.  


TARGET:  

The target for our model is predicting the subreddit category (named 'subredit' in our dataset) based on any text provided in 'post_text'.  In building out my model, I used multiple classification modeling techniques, including a series of APIs from Scikit Learn and TextClassificaton from spaCy. 


#### 3. Model performance on training/test data

In creating my model, I evaluated several different types of classification modeling techniques under two different types of vectorization transformer (CountVectorizer and TF-IDFVectorizer).  Using GridsearchCV via Scikit Learn, each model and transformer was given a series of parameters and scored relative to their train/test split.  

Given the simple nature of tokenization under each transformer, I also decided to employ spaCy and their TextClassification feature to create a model.  Due to the libraries ability to the convulational neural network and training techniques that combine Tokenization, Dependency Parsing, Named Entity Recognition and Similarity comparison, it appealed to me as a more flexible model.

| Model Type                 | Transformer     | Accuracy Score | Iterations | Notes                                                      |
|----------------------------|-----------------|----------------|------------|------------------------------------------------------------|
| Baseline                   | none            | 60.9%          | --         | % of ASOIAF posts                                          |
| spaCy's TextClassification | none            | 91.9%          | 15         | 3.14 Gradient Loss (lowest)                                |
| Multinomial Bayes          | CountVectorizer | 78.2%          | --         | --                                                         |
| RandomForestClassifier     | TF-IDF          | 76.7%          | --         | --                                                         |
| Logistic Regression        | TF-IDF          | 75.3%          | --         | --                                                         |
| Voting Classifier          | TF-IDF          | 72.3%          | --         | 'hard' voting w/ all other Scikit Learn Classifying models |


#### 4. Conclusions

Our best performing model, built on spaCy's TextClassification and which minimized our gradient loss, arrives at an accuracy score of 91.9%, which provided a sizable increase of 31% and out performed any other classification model by 13.7%.  Considering some iterations of our spaCy model saw training scores as high as 94.8% (albeit with a much larger gradient loss), we can assume there is still a lot to be done to improve this model.  Additionally, the model's accuracy increased considerably as more data was passed into it, suggesting that with more time this number could only improve.

The outstanding, relative performance of this model is great, especially considering the amount of overlapping content, but 8 incorrect predictions out of every 100 posts is still too many.  That being said, I would not recommend this to be a production-level model at this time.  

Given more time, to collect data and fine tune some of the hyperparamters, this model could see considerable improvement and could possibly meet production-suitable performance.


#### 5. Next steps

As just previously mentioned, more data would be vital to improving this model.  Additionally, having spent the last week learning the attributes and features of spaCy, there are certainly ways to enhance my model's performance by looking into the following parameters and features of spaCy's TextClassification:

- Adam solver
- Minibatches and compounding parameters
- Dropout rate and decay
- L2 regularization
- stop words application
- text pretraining 

There is a lot of potential here with this model and given more time to learn what occurs in the 'black box' of spaCy's convolutional neural network, how to tune it, and to collect more data to train the model, I could find great success in predicting subreddit post origin.