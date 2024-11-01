from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup as bs
from itertools import islice
import sbert_crossencoder
import sbert_biencoder
import argparse as ap
import string
from tqdm import tqdm
import json
import csv
import re

pt_biencoder_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
pt_cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Provide Doc Collection, QREL, then  topic files
def __main__():
    parser = ap.ArgumentParser('SBert Bi-Encoder & Cross-Encoder IR')
    parser.add_argument('answers_path', type=str, help='Path to Answers File')
    parser.add_argument('topics_1_path', type=str, help='Path to Topics 1 File for Train, Validation, and Test set.')
    parser.add_argument('qrel_path', type=str, help='Path to Qrels File Corresponding to Topics 1')
    parser.add_argument('--extended_topics', type=str, help='Paths to multiple topics file intended for retrieval.', nargs='*')
    args = parser.parse_args()

    answers = read_collection(args.answers_path)
    train_val_test_topics = load_topic_file(args.topics_1_path)
    train_qrel, validation_qrel, test_qrel = split_qrel_dict(args.qrel_path, split=0.9)

    bi_encoder = SentenceTransformer(pt_biencoder_name)
    # cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2')

    test_query_tokens = generate_query_tokens(train_val_test_topics)
    test_query_tokens = {qid:query for qid, query in tqdm(test_query_tokens.items(), desc='Filtering Test Queries...') if qid in test_qrel}

    biencoder_results = sbert_biencoder.retrieve(bi_encoder, test_query_tokens, answers)
    retrieval_to_file(biencoder_results, model='bi', num=1)

    # cross_encoder_results = sbert_cross_encoder.retrieve(cross_encoder, bi_encoder, test_query_tokens, answers)
    # retrieval_to_file(cross_encoder_results, model='cross', num=1)


    #
    # for topic_index, topic_file in topic_file_paths:
    #     sbert_biencoder.retrieve(topic_index, topic_file)
    #     sbert_crossencoder.retrieve(topic_index, topic_file)
    #
    # pass

def split_qrel_dict(qrel_filepath:str, split: float = 0.9) -> tuple[dict, dict, dict]:
    qrel_dict: dict[str, dict[str, int]] = read_qrel_file(qrel_filepath)
    query_count: int = len(qrel_dict)
    train_set_count: int = int(query_count * split)
    val_set_count: int = int((query_count - train_set_count) / 2)
    train_qrel: dict[str, list] = dict(islice(qrel_dict.items(), train_set_count))
    validation_qrel: dict[str, list] = dict(islice(qrel_dict.items(), train_set_count, train_set_count + val_set_count))
    test_qrel: dict[str, list] = dict(islice(qrel_dict.items(), train_set_count + val_set_count, None))
    return train_qrel, validation_qrel, test_qrel

# Given Topics File Path, Return Query Dictionary [ QID -> (Title, Body, Tags) ]
def load_topic_file(topic_filepath: str) -> dict[str, list[str]]:
    queries_json: list[dict[str, str]] = json.load(open(topic_filepath))
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
    doc_list: list[str, dict[str, str]] = json.load(open(answer_filepath))
    doc_dict: dict[str, str] = {}
    for doc in tqdm(doc_list, desc='Reading Doc Collection...', colour='yellow'):
        doc_dict[doc['Id']] = preprocess_text(doc['Text'])
    return doc_dict

# ! Not Removing Stop Words ATM
def preprocess_text(text_string: str) -> str:
    res_str: str = bs(text_string, "html.parser").get_text(separator=' ')
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation})
    # query['Body'].translate(str.maketrans('', '', string.punctuation))
    return res_str

def generate_query_tokens(queries_dict: dict[str, list[str]]) -> dict[str, str]:
    query_tokens: dict[str, str] = {}
    for query_id in tqdm(queries_dict, desc='Generating Query Tokens...', colour='green'):
        query_tokens[query_id] = "[TITLE]" + queries_dict[query_id][0] + "[BODY]" + queries_dict[query_id][1]
    return query_tokens

def retrieval_to_file(retrieval_results: dict[str, dict[str, float]], model: str, num: int):
    with open(f'result_{model}_{num}.tsv', 'w') as csv:
        csv_writer = csv.writer(csv, delimiter='\t')
        for qid, doc_id_scores  in retrieval_results.items():
            for doc_id, score in doc_id_scores.items():
                csv_writer.writerow([qid, 'q0', doc_id, score])
        csv.close()
    pass

if __name__ == '__main__':
    __main__()