from bs4 import BeautifulSoup as bs
from itertools import islice
import sbert_crossencoder
import sbert_biencoder
import string
import json
import csv
import re

# Provide Doc Collection, QREL's, then  topic files
def __main__():
    topic_file_paths = []
    for topic_index, topic_file in topic_file_paths:
        sbert_biencoder.retrieve(topic_index, topic_file)
        sbert_crossencoder.retrieve(topic_index, topic_file)

    
    pass

def build_train_val_test(qrel_filepath:str, split: float = 0.9) -> tuple[dict, dict, dict]:
    qrel_dict: dict[str, dict[str, int]] = read_qrel_file(qrel_filepath)
    query_count: int = len(qrel_dict)
    train_set_count: int = int(query_count * split)
    val_set_count: int = int((query_count - train_set_count) / 2)
    train_set: dict[str, list] = dict(islice(qrel_dict, stop=train_set_count))
    val_set: dict[str, list] = dict(islice(qrel_dict, start=train_set_count, stop=train_set_count + val_set_count))
    test_set: dict[str, list] = dict(islice(qrel_dict, start=train_set_count + val_set_count, stop=None))
    return train_set, val_set, test_set



# Given Topics File Path, Return Query Dictionary [ QID -> (Title, Body, Tags) ]
def load_topic_file(topic_filepath: str) -> dict[str, list[str]]:
    queries_json: list[dict[str, str]] = json.load(open(topic_filepath))
    queries_dict: dict[str, list[str]] = {}
    for query in queries_json:
        title = preprocess_text(query['Title'])
        body = preprocess_text(query['Body'])
        tags = query['Tags']
        queries_dict[query['Id']] = [title, body, tags]
    return queries_dict

# Given Qrel File Path, Return Qrel Dictionary [ QID -> DocID -> Score ]
def read_qrel_file(qrel_filepath: str) -> dict[str, dict[str, int]]:
    qrel_dict: dict[str, dict[str, int]] = {}
    with open(qrel_filepath, "r") as f:
        reader: csv.reader = csv.reader(f, delimiter='\t', linerterminator='\n', quotechar='"')
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
    doc_list: list[str, dict[str, str]] = json.load(open(answer_filepath))
    doc_dict: dict[str, str] = {}
    for doc in doc_list:
        doc_dict[doc['Id']] = preprocess_text(doc['Text'])
    return doc_dict

# ! Not Removing Stop Words ATM
def preprocess_text(text_string: str) -> str:
    res_str: str = bs(text_string, "html.parser").get_text(separator=' ')                           # Remove HTML
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)                                           # Remove URLs
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)                                             # Remove Unicode
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})     # Remove punctuation UNLESS it is a potential space-like seperator.
    # query['Body'].translate(str.maketrans('', '', string.punctuation))
    return res_str

if __name__ == '__main__':
    __main__()