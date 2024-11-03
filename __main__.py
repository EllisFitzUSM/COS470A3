from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from itertools import islice
import sbert_crossencoder
import sbert_biencoder
from tqdm import tqdm
import argparse as ap
import random
import string
import json
import nltk
import csv
import re

try:
    stopwords = stopwords.words('english')
except:
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

def __main__():
    parser = ap.ArgumentParser('SBert Bi-Encoder & Cross-Encoder IR')
    parser.add_argument('answers_path', type=str, help='Path to Answers File')
    parser.add_argument('qrel_path', type=str, help='Path to Qrels File Corresponding to Topics 1')
    parser.add_argument('topics_1_path', type=str, help='Path to Topics 1 File for Train, Validation, and Test set.')
    parser.add_argument('extended_topics', type=str, help='Paths to multiple topics file intended for retrieval.', nargs='*')
    args = parser.parse_args()

    print(len(args.extended_topics))

    answers = read_collection(args.answers_path)
    train_val_test_topics = load_topic_file(args.topics_1_path)
    train_qrel, validation_qrel, test_qrel = split_qrel_dict(args.qrel_path, split=0.9)

    train_val_test_query_tokens = generate_query_tokens(train_val_test_topics)
    test_query_tokens = {
        qid:query
        for qid, query in train_val_test_query_tokens.items() if qid in test_qrel
    }

    pt_bi, pt_ce, ft_bi, ft_ce = get_models(train_qrel, validation_qrel, train_val_test_query_tokens, answers)

    bi_test_results = sbert_biencoder.retrieve(pt_bi, 'bi', 1, test_query_tokens, answers)
    ce_test_results = sbert_crossencoder.re_rank(pt_ce, 'ce', 1, bi_test_results, test_query_tokens, answers)
    ft_bi_test_results = sbert_biencoder.retrieve(ft_bi, 'bi_ft', 1, test_query_tokens, answers)
    ft_ce_test_results = sbert_crossencoder.re_rank(ft_ce, 'ce_ft', 1, ft_bi_test_results, test_query_tokens, answers)

    for topic_index, topic_file in enumerate(args.extended_topics):
        topics_dict = load_topic_file(topic_file)
        query_tokens = generate_query_tokens(topics_dict)

        bi_test_results = sbert_biencoder.retrieve(pt_bi, 'bi', topic_index+2, query_tokens, answers)
        ce_test_results = sbert_crossencoder.re_rank(pt_ce, 'ce', topic_index+2, bi_test_results, query_tokens, answers)
        ft_bi_test_results = sbert_biencoder.retrieve(ft_bi, 'bi_ft', topic_index+2, query_tokens, answers)
        ft_ce_test_results = sbert_crossencoder.re_rank(ft_ce, 'ce_ft', topic_index+2, ft_bi_test_results, query_tokens, answers)

def get_models(train_qrel, validation_qrel, train_val_test_query_tokens, answers) -> tuple[SentenceTransformer, CrossEncoder, SentenceTransformer, CrossEncoder]:
    pt_bi = sbert_biencoder.get_pretrained()
    pt_ce = sbert_crossencoder.get_pretrained()
    ft_bi = sbert_biencoder.fine_tune(train_qrel, validation_qrel, train_val_test_query_tokens, answers)
    ft_ce = sbert_crossencoder.fine_tune(train_qrel, validation_qrel, train_val_test_query_tokens, answers)
    return pt_bi, pt_ce, ft_bi, ft_ce

def split_qrel_dict(qrel_filepath:str, split: float = 0.9) -> tuple[dict, dict, dict]:
    qrel_dict: dict[str, dict[str, int]] = read_qrel_file(qrel_filepath)
    query_ids = list(qrel_dict.keys())
    random.shuffle(query_ids)
    qrel_dict = {query_id:qrel_dict[query_id] for query_id in query_ids}

    query_count: int = len(qrel_dict)
    train_set_count: int = int(query_count * split)
    val_set_count: int = int((query_count - train_set_count) / 2)

    train_qrel: dict[str, dict[str, int]] = dict(islice(qrel_dict.items(), train_set_count))
    validation_qrel: dict[str, dict[str, int]] = dict(islice(qrel_dict.items(), train_set_count, train_set_count + val_set_count))
    test_qrel: dict[str, dict[str, int]] = dict(islice(qrel_dict.items(), train_set_count + val_set_count, None))

    return train_qrel, validation_qrel, test_qrel

# Given Topics File Path, Return Query Dictionary [ QID -> (Title, Body, Tags) ]
def load_topic_file(topic_filepath: str) -> dict[str, list[str]]:
    queries_json: list[dict[str, str]] = json.load(open(topic_filepath, 'r', encoding='utf-8'))
    queries_dict: dict[str, list[str]] = {}
    for query in tqdm(queries_json, desc='Loading Topics/Queries...', colour='blue'):
        title = preprocess_text(query['Title'])
        body = preprocess_text(query['Body'])
        tags = query['Tags']
        queries_dict[query['Id']] = [title, body, tags]
    return queries_dict

# Given Qrel File Path, Return Qrel Dictionary [ QID -> DocID -> Score ]
def read_qrel_file(qrel_filepath: str) -> dict[str, dict[str, int]]:
    qrel_dict: dict[str, dict[str, int]] = {}
    with open(qrel_filepath, "r") as f:
        reader: csv.reader = csv.reader(f, delimiter='\t', lineterminator='\n', quotechar='"')
        for row in reader:
            query_id = row[0]
            doc_id = row[2]
            score = int(row[3])
            if query_id in qrel_dict:
                qrel_dict[query_id][doc_id] = score
            else:
                qrel_dict[query_id] = {doc_id : score}
    return qrel_dict

# Given Answer/Document File Path, Return Document Dictionary [ DocID -> Text ]
def read_collection(answer_filepath: str) -> dict[str, str]:
    doc_list: list[str, dict[str, str]] = json.load(open(answer_filepath, 'r', encoding='utf-8'))
    doc_dict: dict[str, str] = {}
    for doc in tqdm(doc_list, desc='Reading Doc Collection...', colour='yellow'):
        doc_dict[doc['Id']] = preprocess_text(doc['Text'])
    return doc_dict

def preprocess_text(text_string: str) -> str:
    global stopwords
    res_str: str = bs(text_string, "html.parser").get_text(separator=' ')
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
    res_str = ' '.join([word for word in res_str.split() if word not in stopwords])
    return res_str

def generate_query_tokens(queries_dict: dict[str, list[str]]) -> dict[str, str]:
    query_tokens: dict[str, str] = {}
    for query_id in tqdm(queries_dict, desc='Generating Query Tokens...', colour='green'):
        query_tokens[query_id] = "[TITLE]" + queries_dict[query_id][0] + "[BODY]" + queries_dict[query_id][1]
    return query_tokens

if __name__ == '__main__':
    __main__()