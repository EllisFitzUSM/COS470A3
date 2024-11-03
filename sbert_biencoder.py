# Fine-tuning Bi-encoder
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from collections import OrderedDict
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import torch
import math
import os
os.environ["WANDB_DISABLED"] = "true"

pt_biencoder_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
ft_biencoder_path = './ef_ft_bi'
ft_biencoder_name = 'bi-ef-travel-qa-v1'

def get_id_text_embeddings(model, id_text: dict):
    embeddings = model.encode(list(id_text.values()), show_progress_bar=True)
    keys = list(id_text.keys())
    embeddings_dict = {
        keys[index]: embedding for index, embedding in enumerate(embeddings)
    }
    return embeddings_dict

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

def fine_tune(train_dict, validation_dict, query_tokens, answers):
    bi_encoder = get_pretrained()

    num_epochs = 100 # Reducing this could speed up some processes
    batch_size = 16

    ft_dataset = SentencesDataset(convert_train_dict(train_dict, query_tokens, answers), model=bi_encoder)
    ft_dataloader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)
    ft_loss = losses.CosineSimilarityLoss(model=bi_encoder)
    validation_topics, validation_answers, validation_scores = convert_validation_dict(validation_dict, query_tokens, answers)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(validation_topics, validation_answers, validation_scores, write_csv='ef_bi_epoch.csv')
    warmup_steps = math.ceil(len(ft_dataloader) * num_epochs * 0.1)

    bi_encoder.fit(
        train_objectives=[(ft_dataloader, ft_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        save_best_model=True,
        show_progress_bar=True,
        output_path=ft_biencoder_path
    )
    bi_encoder.save(ft_biencoder_path, ft_biencoder_name)

    return bi_encoder

def get_pretrained():
    bi_encoder = SentenceTransformer(pt_biencoder_name)
    bi_encoder.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return bi_encoder

def retrieve(bi_model, name: str, num: int, topics: dict, answers: dict) -> dict[str, dict[str, float]]:
    topic_embeddings = get_id_text_embeddings(bi_model, topics)
    answer_embeddings = get_id_text_embeddings(bi_model, answers)
    results_dict = OrderedDict()

    similarities = bi_model.similarity(
        list(topic_embeddings.values()),
        list(answer_embeddings.values())
    )

    for topic_index, topic_id in enumerate(tqdm(topics, desc='Retrieving Queries...', colour='red')):
        scores = similarities[topic_index, : ]
        top_100_score_idx = np.argpartition(scores, -100)[-100:]
        top_100_scores = {list(answers.keys())[idx]: scores[idx] for idx in top_100_score_idx}
        top_100_sorted = dict(sorted(top_100_scores.items(), key=lambda x: x[1], reverse=True))
        results_dict[topic_id] = top_100_sorted

    with open(f'result_{name}_{num}.tsv', 'w', encoding='utf-8', newline='') as csvfile:
        for qid, doc_id_scores  in results_dict.items():
            for rank, (doc_id, score_tensor) in enumerate(doc_id_scores.items()):
                score = score_tensor.item()
                csvfile.write('\t'.join([qid, 'Q0', doc_id, str(rank), str(score), f'{name}_{num}']) + '\n')
        csvfile.close()

    return results_dict