## News sentiment analysis

**IDS 703 Final Project**

Team members: Adler Viton & Javier Cervantes


This project aims at comparing models that predict sentiment analysis on financial data. 

Structure:
-  `download_hf.py`: downloads the training data from HuggingFace
-  `download_news.py`: a script that uses google news api to download news articles
-  `prep_financial_phrasebank.py`: contains the logic to prepare the financial phrasebank dataset as well as a few classifier models
-  `real_news_classifiers.ipynb`: is a notebook that contains a walkthrough of the real news data extraction, cleaning, embedding, training and testing
-  `synthetic_news_classifiers.ipynb`: is a notebook that contains a walkthrough of the synthetic news data creation, embedding, training and testing
-  `synthetic_vader_classifiers.ipynb`: is a notebook that contains a walkthrough of the synthetic nltk Vader data creation, embedding, training and testing
-  `requirements.txt`: contains the libraries required to run the code


How to run:
-   first run  `pip install -r requirements.txt` to download dependencies
-   run all cells of `real_news_classifiers.ipynb` to see multiple models applied to our real financial phrase-bank data
-   run all cells of `synthetic_news_classifiers.ipynb` to see how the synthetic news data was created and how the models performed on it
-   run all cells of `synthetic_vader_classifiers.ipynb` to see how the synthetic nltk Vader data was created and how the models performed on it
