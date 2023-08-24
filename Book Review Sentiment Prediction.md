**How is this project a value-add?** 

A model that predicts the sentiment of a customer review would facilitate getting an overview of public opinion on the product as it will save the time an employee would take to read through most of the reviews. The created value is saved time - the employees of a company could use this model to expedite a product testing phase, shortening the go-to-market time, which can drive competitive edge.

This project aims to predict the sentiment of a book review, given a book review dataset. It is a classification problem without class imbalance since the label classes are evenly distributed.

 



<u>**Project Plan:**</u>

 

The current book reviews data set is low-dimensional in its current form: it has 2 columns, 1973 rows in total, does not have any missing data. From printing out the first few rows, the only feature is the "Reviews" column, which is of datatype Object and contains the reviews as strings. For this problem, we will use NLP.

 

**Data Preparation**: For this step, we need to transform the "Reviews" text to numerical feature vectors using the TF-IDF, which calculates how relevant a word is in a document relative to a collection of documents. Because TF-IDF provides an understanding of the context of the text data, applying this technique for our sentiment analysis classification (classifying a review as positive or negative) would be more appropriate compared to other techniques.

 

**Context**: TF-IDF stands for Term Frequency-Inverse Document Frequency, which is a numerical representation used in Natural Language Processing (NLP) to measure the importance of a word in a document relative to a collection of documents - corpus. The TF-IDF score for a term in a document is calculated by multiplying two components: term frequency - how many times a term appears in a document relative to the total number of terms in that document, and inverse document frequency - used to deprioritize terms that appear frequently across multiple documents, penalizing common (stop) words. 

The TF-IDF score highlights terms that are frequent in a specific document but rare across the entire corpus, which are often considered to carry important information. Words that are frequent both in a document and across the entire corpus will have a lower TF-IDF score.

The key difference between vectorizers and word embeddings is that word embeddings capture semantic relationships and word similarity, allowing words with similar meanings to be closer in the embedding space, while vectorizers treat words independently and do not capture such relationships.

There is also a difference of when to use word embeddings vs. vectorizers. The best practice for a small vocabulary full of high-frequency words is to use vectorizers. For larger vocabularies full of low-frequency words, word embeddings are better suited.

 

 

**Model selection**: After transforming the data, I want to implement a Logistic Regression and evaluate the and improve the model guided by the resulting loss and accuracy, using the AUC as the evaluation metric. Although a Keras NN would also be appropriate for this problem, regarding the value-add of the model to the business, speed is an important factor. Therefore, I am choosing a Logistic Regression instead of Keras ANN as my model because it is computationally inexpensive compared to a Neural Network.

Neural Networks employ supervised learning techniques – for example they use feed forward and backpropagation along with functions to guide their next predictions during training. However, Neural Networks may be better suited for more complex problems. For example, Logistic Regression has one weighted function that can be used as a model itself to make predictions, whereas Neural Networks use many functions so that the combination of them all is nonlinear (activation function), which allows for more flexibility because data can be modeled with more complex weighted functions and nonlinear relationships.  

 

**Model Development**: Before training, the plan is to split the data set into training and test sets stratified by the label, so that the test set reflects the proportions of the whole data. Afterwards, I will create the TF-IDF Vectorizer with no parameters at first and transform the “Review” feature into numerical vectors, saving them to new variables. Next comes training the Logistic Regression model without specifying the regularization parameter for now.

 

**Evaluation**: After implementing the initial model, the plan is to evaluate the model on the transformed test data based on the AUC and optimize the model using the loss and accuracy scores for determining the optimal parameters for the TF-IDF Vectorizer and Logistic Regression.

 

**Parameter Tuning & Optimization**: First, I would test which value for minimum document frequency drives the highest AUC on test data, which I will use to pass to the final implementation of the vectorizer. Adjusting the regularization hyperparameter C will change the model's log loss, so I will select the value that causes minimal loss and pass it to the final implementation of the model. As the final step, the plan is to compare the AUC of the initial model with the AUC of the optimized model, which will, hopefully, improve.