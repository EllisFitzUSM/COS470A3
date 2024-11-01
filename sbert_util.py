from bs4 import BeautifulSoup as bs
import string
import json
import csv
import re

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
        reader: csv.reader = csv.reader(f, delimiter='\t', lineterminator='\n', quotechar='"')
        for row in reader:
            query_id = row[0]
            doc_id = row[2]
            score = int(row[3])
            if query_id in qrel_dict:
                qrel_dict[query_id][doc_id] = score
            else:
                qrel_dict[query_id] = {doc_id : score}
        f.close()
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