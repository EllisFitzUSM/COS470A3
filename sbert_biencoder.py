# Fine-tuning Bi-encoder
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from itertools import islice
from sbert_util import load_topic_file, read_collection, read_qrel_file
import torch
from tqdm import tqdm
import math
import random
import os
os.environ["WANDB_DISABLED"] = "true"

def get_id_text_embeddings(model, id_text: dict):
    embeddings = {}
    for id, text in tqdm(id_text.items(), desc='Calculating Embeddings...'):
        embeddings[id] = model.encode(text)
    return embeddings

# Convert the Training Dict
def convert_train_dict(train_qrel: dict, query_tokens: dict, answers: dict) -> list[InputExample]:
    train_set: list[InputExample] = []

    for topic_id in train_qrel:
        topic = query_tokens[topic_id]
        relevant_answers = train_qrel.get(topic_id, {})

        for answer_id, score in relevant_answers.items():
            answer = answers[answer_id]
            if score > 1:
                train_set.append(InputExample(texts=[topic, answer], label=1.0))
            else:
                train_set.append(InputExample(texts=[topic, answer], label=0.0))

    return train_set

def convert_validation_dict(validation_dict, query_tokens: dict, answers: dict) -> tuple[list, list, list]:
    validation_set_topics = []
    validation_set_answers = []
    validation_set_scores = []

    for topic_id in validation_dict:
        topic = query_tokens[topic_id]
        relevant_answers = validation_dict.get(topic_id, {})

        for answer_id, score in relevant_answers.items():
            answer = answers[answer_id]
            if score > 1.0:
                label = 1.0
            elif score == 1.0:
                label = 0.5
            else:
                label = 0.0

            validation_set_topics.append(topic)
            validation_set_answers.append(answer)
            validation_set_scores.append(label)

    return validation_set_topics, validation_set_answers, validation_set_scores


# Uses the posts file, topic file(s) and qrel file(s) to build our training and evaluation sets.
# def process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic):
#     train_samples = []
#     evaluator_samples_1 = []
#     evaluator_samples_2 = []
#     evaluator_samples_score = []
#
#     # Build Training set
#     for topic_id in train_dic_qrel:
#         question = queries[topic_id]
#         dic_answer_id = train_dic_qrel.get(topic_id, {})
#
#         for answer_id in dic_answer_id:
#             score = dic_answer_id[answer_id]
#             answer = collection_dic[answer_id]
#             if score > 1:
#                 train_samples.append(InputExample(texts=[question, answer], label=1.0))
#             else:
#                 train_samples.append(InputExample(texts=[question, answer], label=0.0))
#
#     for topic_id in val_dic_qrel:
#         question = queries[topic_id]
#         dic_answer_id = val_dic_qrel.get(topic_id, {})
#
#         for answer_id in dic_answer_id:
#             score = dic_answer_id[answer_id]
#             answer = collection_dic[answer_id]
#             if score > 1:
#                 label = 1.0
#             elif score == 1:
#                 label = 0.5
#             else:
#                 label = 0.0
#             evaluator_samples_1.append(question)
#             evaluator_samples_2.append(answer)
#             evaluator_samples_score.append(label)
#
#     return train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score



def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    return {key: d[key] for key in keys}

def split_train_validation(qrels, ratio=0.9):
    # Using items() + len() + list slicing
    # Split dictionary by half
    n = len(qrels)
    n_split = int(n * ratio)
    qrels = shuffle_dict(qrels)
    train = dict(islice(qrels.items(), n_split))
    validation = dict(islice(qrels.items(), n_split, None))

    return train, validation

def fine_tune(model: SentenceTransformer, train_dict, validation_dict, query_tokens: dict, answers: dict):

    num_epochs = 100 # Reducing this could speed up some processes
    batch_size = 16

    my_model_name = 'ef-travel-qa-v1'
    my_model_path = './ef_ft_bi_2024'

    ft_dataset = SentencesDataSet(convert_train_dict(train_dict, query_tokens, answers))
    ft_dataloader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)
    ft_loss = losses.CosineSimilarityLoss(model=model)
    validation_topics, validation_answers, validation_scores = convert_validation_dict(validation_dict, topics, answers)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(validation_topics, validation_answers, validation_scores, write_csv='ef_bi_epoch.csv')
    warmup_steps = math.ceil(len(ft_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(ft_dataloader, ft_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        save_best_model=True,
        show_progress_bar=True,
        output_path=my_model_path
    )

    model.save(my_model_path, my_model_name)



def train(model):
    # reading queries and collection
    dic_topics = load_topic_file("topics_1.json")
    queries = {}
    for query_id in dic_topics:
        queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
    qrel = read_qrel_file("qrel_1.tsv")
    collection_dic = read_collection('Answers.json')
    train_dic_qrel, val_dic_qrel = split_train_validation(qrel)

    num_epochs = 100
    batch_size = 16

    # Rename this when training the model and keep track of results
    MODEL = "SAVED_MODEL_NAME"

    # Creating train and val dataset
    train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score = process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic)

    convert_train_dict(train_dic_qrel, queries, collection_dic)
    validation_topics, validation_answers, validation_scores = convert_validation_dict(val_dic_qrel, queries, collection_dic)

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(evaluator_samples_1, evaluator_samples_2, evaluator_samples_score, write_csv="evaluation-epoch.csv")
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # add evaluator to the model fit function
    model.fit(
        train_objectives =[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        save_best_model=True,
        show_progress_bar=True,
        output_path=MODEL
    )


def retrieve(model, topics: dict, answers: dict) -> dict[str, dict[str, float]]:
    topic_embeddings = get_id_text_embeddings(model, topics)
    answer_embeddings = get_id_text_embeddings(model, answers)
    results_dict = {}

    print('hi')
    similarities = model.similarity(topic_embeddings, answer_embeddings)
    print('bye')
    for topic_index, topic_id in tqdm(enumerate(topics), desc='Retrieving Queries...', colour='red'):
        results_dict[topic_id] = {}
        for answer_index, answer_id in enumerate(answers):
            similarity_score = similarities[topic_index, answer_index]
            results_dict[topic_id][answer_id] = similarity_score

        results_dict[topic_id] = sorted(results_dict[topic_id], key=lambda x: results_dict[topic_id][x], reverse=True)[:100]

    return results_dict

# model = SentenceTransformer('all-MiniLM-L6-v2')
def stuff():
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    train(model)