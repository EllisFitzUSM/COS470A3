# Fine-tuning Cross-encoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import CrossEncoder, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import sbert_biencoder
from tqdm import tqdm
import torch
import math

pt_crossencoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
ft_crossencoder_path = './ef_ft_ce'
ft_crossencoder_name = 'ce-ef-travel-qa-v1'

def re_rank(cross_encoder, name: str, num: int, rankings: dict[str, dict[str, float]], query_tokens: dict, answers: dict) -> dict[str, dict[str, float]]:
    re_ranked = {}
    for query_id, answer_to_scores in tqdm(rankings.items(), desc='Re-Ranking BiEncoder w/ CrossEncoder...'):

        query = query_tokens[query_id]
        matchings = [[answer_id, query, answers[answer_id]] for answer_id, score in answer_to_scores.items()]

        cross_scores = cross_encoder.predict([matching[1:] for matching in matchings])

        score_dict = {
            matchings[index][0]:cross_score
            for index, cross_score in enumerate(cross_scores)
        }

        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
        re_ranked[query_id] = score_dict

    with open(f'result_{name}_{num}.tsv', 'w', encoding='utf-8', newline='') as csvfile:
        for qid, doc_id_scores  in re_ranked.items():
            for rank, (doc_id, score_tensor) in enumerate(doc_id_scores.items()):
                score = score_tensor.item()
                csvfile.write('\t'.join([qid, 'Q0', doc_id, str(rank), str(score), f'{name}_{num}']) + '\n')
        csvfile.close()

    return re_ranked

def fine_tune(train_dict, validation_dict, query_tokens, answers):
    cross_encoder = get_pretrained()

    train_samples = convert_train_dict(train_dict, query_tokens, answers)
    valid_samples = convert_validation_dict(validation_dict, query_tokens, answers)

    num_epochs = 2
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)
    # During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CERerankingEvaluator(valid_samples, name='train-eval')
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    # train_loss = losses.MultipleNegativesRankingLoss(model=cross_encoder)
    cross_encoder.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=ft_crossencoder_path,
        save_best_model=True
    )

    cross_encoder.save(ft_crossencoder_path)
    return cross_encoder

def get_pretrained():
    cross_encoder = CrossEncoder(pt_crossencoder_name, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    tokens = ["[TITLE]", "[BODY]"]
    cross_encoder.tokenizer.add_tokens(tokens, special_tokens=True)
    cross_encoder.model.resize_token_embeddings(len(cross_encoder.tokenizer))
    return cross_encoder

# Convert the Training Dict
def convert_train_dict(train_qrel: dict, query_tokens: dict, answers: dict) -> list[InputExample]:
    train_set: list[InputExample] = []

    for topic_id in train_qrel:
        topic = query_tokens[topic_id]
        relevant_answers = train_qrel.get(topic_id, {})

        for answer_id, score in relevant_answers.items():
            answer = answers[answer_id]
            label = score
            if label >= 1:
                label = 1
            train_set.append(InputExample(texts=[topic, answer], label=label))

    return train_set

def convert_validation_dict(validation_dict, query_tokens: dict, answers: dict) -> tuple[list, list, list]:
    validation_samples: dict[str, str] = {}

    for topic_id in validation_dict:
        topic = query_tokens[topic_id]
        relevant_answers = validation_dict.get(topic_id, {})

        for answer_id, score in relevant_answers.items():
            answer = answers[answer_id]
            label = score
            if topic_id not in validation_samples:
                validation_samples[topic_id] = {'query': topic, 'positive': set(), 'negative': set()}

            if label == 0:
                label = 'negative'
            else:
                label = 'positive'
            validation_samples[topic_id][label].add(answer)

    return validation_samples

