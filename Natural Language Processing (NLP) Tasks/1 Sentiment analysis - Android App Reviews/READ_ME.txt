# Sentiment Analysis of Android App Reviews

This project demonstrates the development of a robust sentiment analysis system for classifying Android app reviews 
(retrieved, in this example, from Amazon) as positive, negative, or neutral.  This is an NLP exercise showcasing the
power of machine learning in understanding customer feedback.

## Overview

This system utilises a classifier trained to recognise the nuances of written language after vectorisation. This 
allows for accurate prediction of the sentiment expressed in app reviews.  Such a system can be invaluable for 
companies seeking to understand customer opinions by classifying, quantifying, and analysing review data.  By 
understanding customer sentiment, businesses can gain valuable insights into product perception, identify areas 
for improvement, and make data-driven decisions.

## Project Structure

The project code is organised into two Jupyter Notebooks written in Python:

1. **`CW2 - Training.ipynb`**: This notebook focuses on the pre-processing of the textual data (the app reviews). It 
includes steps for cleaning and preparing the text data for analysis.  Crucially, this notebook also covers the 
training and performance comparison of various classification algorithms.  This allows for a thorough evaluation of 
different models to determine the most effective approach for sentiment classification in this context.

2. **`CW2 - Testing.ipynb`**: This notebook contains the code for testing the best-performing classifier (identified 
in the training notebook) on a separate set of test reviews.  It also includes code for visualising the results of the
sentiment analysis.  A key feature of this notebook is the identification of the most popular Android app based on the 
sentiment expressed in its reviews.  This provides a practical demonstration of how sentiment analysis can be used to 
identify trends and preferences.

## Key Features and Technologies

* **Natural Language Processing (NLP):** Core techniques for text processing and analysis.
* **Sentiment Analysis:**  Classification of text into positive, negative, or neutral categories.
* **Machine Learning:** Training classifiers to recognise sentiment patterns.
* **Jupyter Notebooks:**  Interactive environment for code development and documentation.
* **Python:** The programming language used for this project.
* **[List any specific libraries used, e.g., scikit-learn, NLTK, spaCy, etc.]**:  
* **[Mention the dataset used, if applicable.  If it's a custom dataset, briefly describe how it was created/obtained.]:**

## Future Work 

* Explore other classification algorithms.
* Implement a user interface for easier interaction.
* Expand the analysis to include other app stores or review platforms.