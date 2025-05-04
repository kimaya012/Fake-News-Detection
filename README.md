# Fake-News-Detection

The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not. The distribution of fake news can potentially have highly adverse effects on people and culture. This project involves building and training a model to classify news as fake news or not using a diverse dataset of news articles. We have used four techniques to determine the results of the model.

1. Logistic Regression
2. Decision Tree Classifier
3. Gradient Boost Classifier
4. Random Forest Classifier
- (Baseline: Multinomial Naive Bayes)

## Project Overview
 
 Fake news has become a significant issue in today's digital age, where information spreads rapidly through various online platforms. This project leverages machine learning algorithms to automatically determine the authenticity of news articles, providing a valuable tool to combat misinformation.
 
 ## Dataset
 This project uses the Fake and Real News Dataset by Clément Bisaillon, available on Kaggle.

Source: [Click](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

It is a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
 - True: Genuine news articles
 - False: Fake or fabricated news articles

## Dependencies
 
 Before running the code, make sure you have the following libraries and packages installed:
 
 - Python 3
 - Scikit-learn
 - Pandas
 - Numpy
 - Seaborn
 - Matplotlib
 - Regular Expression
 
 You can install these dependencies using pip:
 
 ```bash
 pip install pandas
 pip install numpy
 pip install matplotlib
 pip install sklearn
 pip install seaborn 
 pip install re 
 ```

## Results
 
 We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files.
