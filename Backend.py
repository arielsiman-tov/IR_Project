from collections import Counter
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pickle
from google.cloud import storage
import math
import numpy as np
from inverted_index_gcp import *
from heapq import heappop, heappush, heapify
from threading import Thread
import re
import inflect
# from annoy import AnnoyIndex

import time
nltk.download('stopwords')

# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

bucket_name = "209234103_final"


# Function to download the index from the storage
def download_index_from_storage(file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    return pickle.loads(contents)


# Import inverted dicts
PageRank_dict = download_index_from_storage("PageRank/PageRank_index.pkl")
doc_to_title_dict = download_index_from_storage("doc_to_title_dict/doc_to_title_dict_index.pkl")
PageViews_dict = download_index_from_storage("PageViews/PageViews_index.pkl")

# Import inverted indexes stemming
# Index_Title = download_index_from_storage("Title/Title_index.pkl")
# Index_Body = download_index_from_storage("Body/Body_index.pkl")
# Index_Title_BM25 = download_index_from_storage("Title_BM25/Title_BM25_index.pkl")
# Index_Body_BM25 = download_index_from_storage("Body_BM25/Body_BM25_index.pkl")

# Import inverted indexes no stemming
Index_Title = download_index_from_storage("Title_no_stem/Title_no_stem_index.pkl")
Index_Body = download_index_from_storage("Body_no_stem/Body_no_stem_index.pkl")
Index_Title_BM25 = download_index_from_storage("Title_no_stem_BM25/Title_no_stem_BM25_index.pkl")
Index_Body_BM25 = download_index_from_storage("Body_no_stem_BM25/Body_no_stem_BM25_index.pkl")

# Import inverted indexes lemmatization
# Index_Title = download_index_from_storage("Title_lemm/Title_lemm_index.pkl")
# Index_Body = download_index_from_storage("Body_lemm/Body_lemm_index.pkl")
# Index_Title_BM25 = download_index_from_storage("Title_lemm_BM25/Title_lemm_BM25_index.pkl")
# Index_Body_BM25 = download_index_from_storage("Body_lemm_BM25/Body_lemm_BM25_index.pkl")

# Import Word2Vec model
# Word2Vec_model = download_index_from_storage("Word2VEC_Model/Word2VEC_Model_index.pkl"



# Function to get the HTML pattern
def get_html_pattern():
    return r"<[^>]+>"


# Function to get the date pattern
def get_date_pattern():
    return r"(?:Jan(?:uary)?|Mar(?:ch)?|May|Jul(?:y)?|Aug(?:ust)?|Oct(?:ober)|Dec(?:ember)?)\s(?:0[1-9]|1[0-9]|2[0-9]|3[01]|[1-9]),\s\d{4}|(?:Apr(?:il)?|Jun(?:e)?|Sep(?:tember)?|Nov(?:ember)?)\s(?:0[1-9]|1[0-9]|2[0-9]|3[0]|[1-9]),\s\d{4}|(?:Feb(?:ruary)?)\s(?:0[1-9]|1[0-9]|2[0-9]|[1-9]),\s\d{4}|(?:0[1-9]|1[0-9]|2[0-9]|3[01]|[1-9])\s(?:Jan(?:uary)?|Mar(?:ch)?|May|Jul(?:y)?|Aug(?:ust)?|Oct(?:ober)|Dec(?:ember)?),\s\d{4}|(?:0[1-9]|1[0-9]|2[0-9]|3[01]|[1-9])\s(?:Jan(?:uary)?|Mar(?:ch)?|May|Jul(?:y)?|Aug(?:ust)?|Oct(?:ober)|Dec(?:ember)?)\s\d{4}|(?:0[1-9]|1[0-9]|2[0-9]|3[0]|[1-9])\s(?:Apr(?:il)?|Jun(?:e)?|Sep(?:tember)?|Nov(?:ember)?)\s\d{4}|(?:0[1-9]|1[0-9]|2[0-9]|[1-9])\s(?:Feb(?:ruary)?)\s\d{4}"


# Function to get the time pattern
def get_time_pattern():
    return r"((?:[0-1]\d[0-5]\d|[2][0-4][0-5]\d|[0-1]\d\.[0-5]\d|[2][0-4]\.[0-5]\d)(?:[APM]{2}|[apm.]{4}))|(\b(?:[0-1]\d|[2][0-4]|\d{1})(?::[0-5]\d){2}\b)"


# Function to get the percent pattern
def get_percent_pattern():
    return r"\d+(\.\d+)?%"


# Function to get the number pattern
def get_number_pattern():
    return r"\d+"


# Function to get the word pattern
def get_word_pattern():
    return r"\b(?<![-,:=\+\$\w])\d?[A-Za-z\'-]+"


RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+)
    # everything else
    |(?P<OTHER>.))""", re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize_query(text):
    """
    Tokenizes the input text by extracting words and numbers, converting them to lowercase, removing stopwords, and returning a list of tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of tokens extracted from the input text.
    """
        
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = ' '.join(tokens)
    tokens = [v for match in RE_TOKENIZE.finditer(tokens)
              for k, v in match.groupdict().items()
              if v is not None and k in ['WORD']]
    tokens = [token for token in tokens if token not in all_stopwords]

    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(token) for token in tokens]
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]

    numbers = [v for match in RE_TOKENIZE.finditer(text)
               for k, v in match.groupdict().items()
               if v is not None and k in ['NUMBER']]
    
    # extract numbers and convert them to words
    if len(numbers) != 0:
        p = inflect.engine()
        number_words = [p.number_to_words(int(number)) for number in numbers]
        all_words = numbers
        for number_word in number_words:
            all_words.extend(number_word.split())

        combined_list = []
        combined_list.extend(tokens)
        combined_list.extend(all_words)
        tokens = combined_list
        tokens = [token for token in tokens if token not in all_stopwords]
    return tokens


def get_top_N_dict(scores_dict, N):
    """
    Retrieves the top N documents based on their scores from a dictionary of scores.

    Args:
        scores_dict (dict): A dictionary containing document scores.
        N (int): The number of top documents to retrieve.

    Returns:
        dict: A dictionary containing the top N documents and their scores.
    """
        
    result_dict = {}
    heap = []
    heapify(heap)
    for key, value in scores_dict.items():
        heappush(heap, (-1 * value, key))
    counter = 0
    while heap and counter < N:
    # while heap:
        score, doc_id = heappop(heap)
        result_dict[doc_id] = score * -1
        counter += 1
    return result_dict


def get_top_N(scores_dict, N):
    """
    Retrieves the top N documents based on their scores from a dictionary of scores.

    Args:
        scores_dict (dict): A dictionary containing document scores.
        N (int): The number of top documents to retrieve.

    Returns:
        list: A list of tuples containing the top N documents and their scores.
    """
        
    result = []
    heap = []
    heapify(heap)
    for key, value in scores_dict.items():
        heappush(heap, (-1 * value, key))
    counter = 0
    while heap and counter < N:
        score, doc_id = heappop(heap)
        result.append((doc_id, score * -1))
        counter += 1
    return result


def get_doc_title(scores):
    """
    Retrieves document titles corresponding to the given scores.

    Args:
        scores (list): A list of tuples containing document IDs and their scores.

    Returns:
        list: A list of tuples containing document IDs and their titles.
    """

    result = []
    for doc_id, score in scores:
        try:
            result.append((str(doc_id), doc_to_title_dict[doc_id]))
        except KeyError:
            continue
    return result


def cosSim(candidate_docs, index, query_tf_idf):
    """
    Calculates the cosine similarity scores between candidate documents and the query based on TF-IDF scores.

    Args:
        candidate_docs (dict): A dictionary containing candidate documents and their TF-IDF scores.
        index (dict): The index containing document information.
        query_tf_idf (dict): A dictionary containing TF-IDF scores for query terms.

    Returns:
        dict: A dictionary containing the cosine similarity scores for candidate documents.
    """
        
    cosSim_scores = {}
    for doc_id in candidate_docs:
        try:
            denominator = math.sqrt(index.weights_square[doc_id] * sum([math.pow(score, 2) for score in query_tf_idf.values()]))
        except KeyError:
            cosSim_scores[doc_id] = 0
        if denominator != 0:
            cosSim_scores[doc_id] = candidate_docs[doc_id] / denominator
        else:
            cosSim_scores[doc_id] = 0
    return cosSim_scores


def tf_idf_query(query, index):
    """
    Calculates TF-IDF scores for the terms in the query.

    Args:
        query (list): A list of terms in the query.
        index (dict): The index containing document information.

    Returns:
        dict: A dictionary containing TF-IDF scores for the query terms.
    """
        
    query_scores = {}
    query_tf = Counter(query)
    for term in np.unique(query):
        try:
            query_scores[term] = (query_tf[term] / len(query)) * math.log(index.N / index.df[term], 2)
        except KeyError:
            continue
    return query_scores


def BM25_idf_query(query, index):
    """
    Calculates IDF scores for the terms in the query using the BM25 algorithm.

    Args:
        query (list): A list of terms in the query.
        index (dict): The index containing document information.

    Returns:
        dict: A dictionary containing IDF(based on BM25) scores for the query terms.
    """
        
    idf = {}
    N = index.N 
    for token in np.unique(query):
        try:
            n = index.df.get(token, 0)  
            idf[token] = np.log(((N - n + 0.5) / (n + 0.5)) + 1)
        except KeyError:
            continue
    return idf


# def candidate_tf_idf(query, index, query_idf):
#     candidate_docs = {}
#     for term in np.unique(query):
#         try:
#             posting_list = index.read_a_posting_list(".", term, bucket_name)
#             for doc_id, tf in posting_list:
#                     weight = (tf / index.doc_len[doc_id]) * math.log((index.N / index.df[term]), 2) * query_idf[term]
#                     candidate_docs[doc_id] = candidate_docs.get(doc_id, 0) + weight
#         except KeyError:
#             continue
#     return candidate_docs

def candidate_tf_idf(query, index, query_idf):
    """
    Generates candidate documents and their scores using TF-IDF weighting.

    Args:
        query (list): A list of terms in the query.
        index (dict): The index containing document information.
        query_idf (dict): A dictionary containing IDF scores for the query terms.

    Returns:
        dict: A dictionary containing candidate documents and their scores.
    """
        
    candidate_docs = {}
    if index == Index_Title:
        threshold = 0
    else:
        threshold = 5
    max_documents = 50000
    query = sorted(query, key=lambda term: index.idf.get(term, 0), reverse=True)
    threads = []

    for term in np.unique(query):
        thread = Thread(target=lambda: process_term_tf_idf(term, index, query_idf, candidate_docs, threshold, max_documents))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return candidate_docs

def process_term_tf_idf(term, index, query_idf, candidate_docs, threshold, max_documents):
    """
    Processes a term for TF-IDF scoring, updating the candidate documents with TF-IDF weighted scores.

    Args:
        term (str): The term to process.
        index (object): The index object.
        query_idf (dict): A dictionary containing IDF scores for query terms.
        candidate_docs (dict): A dictionary containing candidate documents and their scores.
        threshold (int): The minimum TF threshold for considering a term in a document.
        max_documents (int): The maximum number of documents to consider.

    Returns:
        None
    """
        
    posting_list = fetch_posting_list(index, term, bucket_name)
    for doc_id, tf in posting_list:
        if tf > threshold and len(candidate_docs) < max_documents:
            weight = (tf / index.doc_len[doc_id]) * math.log((index.N / index.df[term]), 2) * query_idf[term]
            candidate_docs[doc_id] = candidate_docs.get(doc_id, 0) + weight
        else:
            break


# def candidate_doc_BM25(query, index, query_idf):
#     candidate_docs = {}
#     for term in np.unique(query):
#         try:
#             posting_list = index.read_a_posting_list(".", term, bucket_name)
#             for doc_id, tf in posting_list:
#                 weight = query_idf[term] * (tf * index.weights_square[doc_id][0]) / (
#                         tf + index.weights_square[doc_id][1])
#                 candidate_docs[doc_id] = candidate_docs.get(doc_id, 0) + weight
#         except KeyError:
#             continue
#     return candidate_docs
        
def fetch_posting_list(index, term, bucket_name):
    """
    Fetches the posting list for a given term from the index.

    Args:
        index (object): The index object.
        term (str): The term to fetch the posting list for.
        bucket_name (str): The name of the bucket where the posting list is stored.

    Returns:
        list: The posting list for the given term sorted by TF-IDF scores in descending order.
    """
        
    try:
        posting_list = index.read_a_posting_list(".", term, bucket_name)
        posting_list.sort(key=lambda entry: entry[1], reverse=True)
        return posting_list
    except KeyError:
        return []       
    
def candidate_doc_BM25(query, index, query_idf):
    """
    Generates candidate documents and their scores using BM25 weighting.

    Args:
        query (list): A list of terms in the query.
        index (dict): The index containing document information.
        query_idf (dict): A dictionary containing IDF scores for the query terms.

    Returns:
        dict: A dictionary containing candidate documents and their scores.
    """
        
    candidate_docs = {}
    if index == Index_Title_BM25:
        threshold = 0
    else:
        threshold = 5
    max_documents = 50000
    query = sorted(query, key=lambda term: index.idf.get(term, 0), reverse=True)
    threads = []

    for term in np.unique(query):
        thread = Thread(target=lambda: process_term_BM25(term, index, query_idf, candidate_docs, threshold, max_documents))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return candidate_docs


def process_term_BM25(term, index, query_idf, candidate_docs, threshold, max_documents):
    """
    Processes a term for BM25 scoring, updating the candidate documents with BM25 weighted scores.

    Args:
        term (str): The term to process.
        index (object): The index object.
        query_idf (dict): A dictionary containing IDF scores for query terms.
        candidate_docs (dict): A dictionary containing candidate documents and their scores.
        threshold (int): The minimum TF threshold for considering a term in a document.
        max_documents (int): The maximum number of documents to consider.

    Returns:
        None
    """
        
    posting_list = fetch_posting_list(index, term, bucket_name)
    for doc_id, tf in posting_list:
        if tf > threshold and len(candidate_docs) < max_documents:
            weight = query_idf[term] * (tf * index.weights_square[doc_id][0]) / (
                    tf + index.weights_square[doc_id][1])
            candidate_docs[doc_id] = candidate_docs.get(doc_id, 0) + weight
        else:
            break


def search_tf_idf(query, index):
    """
    Searches for documents based on the given query using TF-IDF scoring.

    Args:
        query (str): The search query.
        index (dict): The index containing document information.

    Returns:
        dict: A dictionary containing the top N documents and their scores based on TF-IDF scoring.
    """
    ## Tokenize the query    
    query = tokenize_query(query)
    ## Calculate the TF-IDF scores for the query
    query_tf_idf = tf_idf_query(query, index)
    ## Generate candidate documents based on the query
    candidate_docs = candidate_tf_idf(query, index, query_tf_idf)
    if candidate_docs is None:
        return []
    ## Calculate the cosine similarity scores between candidate documents and the query
    scores_dict = cosSim(candidate_docs, index, query_tf_idf)
    ## Retrieve the top 5000 documents based on their scores
    result_rank = get_top_N_dict(scores_dict, 5000)
    return result_rank


def search_BM25(query, index):
    """
    Searches for documents based on the given query using BM25 scoring.

    Args:
        query (str): The search query.
        index (dict): The index containing document information.

    Returns:
        dict: A dictionary containing the top N documents and their scores based on BM25 scoring.
    """
    ## Tokenize the query
    query = tokenize_query(query)
    
    # try:
    #     query_new = (Word2Vec_model.most_similar(query, topn=5))
    #     query_new = [word[0] for word in query_new]
    #     query = list(set(query + query_new))
    #     stemmer = PorterStemmer()
    #     query = [stemmer.stem(token) for token in query]
    # except KeyError:
    #     pass    

    ## Calculate the IDF scores for the query using BM25 algorithm
    query_idf = BM25_idf_query(query, index)
    ## Generate candidate documents based on the query
    candidate_docs = candidate_doc_BM25(query, index, query_idf)
    if candidate_docs is None:
        return []
    ## Retrieve the top 5000 documents based on their scores
    result_rank = get_top_N_dict(candidate_docs, 5000)
    return result_rank


def search_weights(query):
    """
    Searches for documents based on the given query using a weighted combination of TF-IDF scores for both body and title fields,
    along with PageRank and PageView weights.

    Args:
        query (str): The search query.

    Returns:
        dict: A dictionary containing the top N documents and their titles based on the weighted ranking.
    """
    ## Define the weights for the different ranking components
    Body_weight = 12
    Title_weight = 4
    PageRank_weight = 2
    PageView_weight = 4
    results = {}

    ## Create threads to search and rank documents based on TF-IDF scores for both body and title fields
    thread_body = Thread(target=search_and_rank_tf_idf, args=(query, Index_Body, results, "Body_rank"))
    thread_title = Thread(target=search_and_rank_tf_idf, args=(query, Index_Title, results, "Title_rank"))

    ## Start the threads
    thread_body.start()
    thread_title.start()

    ## Wait for the threads to finish
    thread_body.join()
    thread_title.join()

    ## Retrieve the results from the threads
    Body_rank = results["Body_rank"]
    Title_rank = results["Title_rank"]

    # Body_rank = search_Body(query,Index_Body)
    # Title_rank = search_Title(query,Index_Title)

    ## Combine the scores from the different ranking components
    docs = np.unique(list(Body_rank.keys()) + list(Title_rank.keys()))
    result = {}
    for d in docs:
        PageRank_value = PageRank_dict.get(d, 1)
        PageView_value = PageViews_dict.get(d, 1)  
        PageRank = np.log2(PageRank_value) * PageRank_weight
        PageView = np.log2(PageView_value) * PageView_weight

        result[d] = (Title_weight * Title_rank.get(d, 0) +
                     Body_weight * Body_rank.get(d, 0) +
                    PageRank + PageView)

    ## Retrieve the top 100 documents based on their scores
    result = get_top_N(result, 100)
    ## Retrieve the titles of the top 100 documents
    result_1 = get_doc_title(result)
    return result_1


def search_weights_BM25(query):
    """
    Searches for documents based on the given query using a weighted combination of BM25 scores for both body and title fields,
    along with PageRank and PageView weights.

    Args:
        query (str): The search query.

    Returns:
        dict: A dictionary containing the top N documents and their titles based on the weighted ranking.
    """
    ## Define the weights for the different ranking components
    BM_25_Body_weight = 12
    BM_25_Title_weight = 4
    PageRank_weight = 2
    PageView_weight = 4
    results = {}

    ## Create threads to search and rank documents based on BM25 scores for both body and title fields
    thread_body = Thread(target=search_and_rank_BM25, args=(query, Index_Body_BM25, results, "Body_rank"))
    thread_title = Thread(target=search_and_rank_BM25, args=(query, Index_Title_BM25, results, "Title_rank"))

    ## Start the threads
    thread_body.start()
    thread_title.start()

    ## Wait for the threads to finish
    thread_body.join()
    thread_title.join()

    ## Retrieve the results from the threads
    Body_rank = results["Body_rank"]
    Title_rank = results["Title_rank"]

    # Body_rank = search_Body(query,Index_Body_BM25)
    # Title_rank = search_Title(query,Index_Title_BM25)

    ## Combine the scores from the different ranking components
    docs = np.unique(list(Body_rank.keys()) + list(Title_rank.keys()))
    result = {}
    for d in docs:
        PageRank_value = PageRank_dict.get(d, 1)
        PageView_value = PageViews_dict.get(d, 1)
        PageRank = np.log2(PageRank_value) * PageRank_weight
        PageView = np.log2(PageView_value) * PageView_weight
    
        result[d] = (BM_25_Title_weight * Title_rank.get(d, 0) +
                    BM_25_Body_weight * Body_rank.get(d, 0) +
                    PageRank + PageView)
        
    ## Retrieve the top 100 documents based on their scores
    result = get_top_N(result, 100)
    ## Retrieve the titles of the top 100 documents
    result_1 = get_doc_title(result)
    return result_1


def search_and_rank_BM25(query, index, result_dict, key):
    """
    Helper function for searching and ranking documents using BM25 algorithm.

    Args:
        query (str): The search query.
        index (dict): The index containing document information.
        result_dict (dict): A dictionary to store search results.
        key (str): The key to store results in the result_dict.
    """
        
    result_dict[key] = search_BM25(query, index)


def search_and_rank_tf_idf(query, index, result_dict, key):
    """
    Helper function for searching and ranking documents using TF-IDF algorithm.

    Args:
        query (str): The search query.
        index (dict): The index containing document information.
        result_dict (dict): A dictionary to store search results.
        key (str): The key to store results in the result_dict.
    """
        
    result_dict[key] = search_tf_idf(query, index)

