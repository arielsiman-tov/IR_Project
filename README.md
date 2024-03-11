#  Information Retrieval Engine

## Authors
* **Tal Klein** 
* **Ariel Siman Tov** 

## Project Overview

This project focuses on developing a retrieval information system based on ranking methods. The core components of this system include the creation of inverted indexes, backend calculation for ranking methods, and a search frontend to facilitate user queries.

## Data

- **Entire Wikipedia Dump**: complete Wikipedia dump from a shared Google Storage bucket.


## Queries and Ranked Results:

The queries and a ranked list of up to 100 relevant results. The ranking is based on BM25 on the body and title of documents, as well as PageRank and pageviews.

See `queries_train.json` for  training queries.



## Files

- **Indexing**

   - **tf_idf_title_index.py**: Creates indexes based on tf-idf for title extraction for each document from the entire corpus, with options for stemming, lemmatization, and no preprocessing.

   - **tf_idf_body_index.py**: Creates indexes based on tf-idf for  text extraction for each document from the entire corpus, with options for stemming, lemmatization, and no preprocessing.

   - **BM25_title_index.py**: Creates indexes based on BM25 for title extraction for each document from the entire corpus, with options for stemming, lemmatization, and no preprocessing.

   - **BM25_body_index.py**: Creates indexes based on BM25 for text extraction for each document from the entire corpus, with options for stemming, lemmatization, and no preprocessing.

   - **Dictionaries** : Dictionaries include information such as page views, PageRank, document-to-title mapping, and a word2vec model.

- **backend.py**: The backend engine is implemented in this file. It contains functions for calculating various ranking methods such as TF-IDF, BM25, PageRank, etc. These ranking methods are essential for determining the relevance of documents to a given query.

- **inverted_index.gcp.py**: This file defines the class for the inverted index. It includes methods for adding documents to the index, updating the index, and calculating relevant features for documents.

- **search_frontend.py**: The search frontend is initiated in this script. It provides an interface for users to input their queries and receive relevant search results. The frontend interacts with the backend engine to perform the necessary calculations and retrieve documents.





