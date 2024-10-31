# Fine-tuning Cross-encoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import math
from sentence_transformers import CrossEncoder, InputExample, losses, evaluation # ?
from torch.utils.data import DataLoader
from sbert_util import load_topic_file, read_collection, read_qrel_file

def retrieve(biencoder_rankings : dict[str, list[str]]):

    pass

def generate_query_tokens(queries_dict: dict[str, list[str]]) -> dict[str, str]:
    query_tokens: dict[str, str] = {}
    for query_id in queries_dict:
        query_tokens[query_id] = "[TITLE]" + queries_dict[query_id][0] + "[BODY]" + queries_dict[query_id][1]
    return query_tokens

queries_dict = load_topic_file("topics_1.json")
query_tokens = generate_query_tokens(queries_dict)
qrel = read_qrel_file("qrel_1.tsv")
collection_dic = read_collection('Answers.json')

## Preparing pairs of training instances
num_topics = len(query_tokens.keys())
number_training_samples = int(num_topics*0.9)

## Preparing the content
counter = 1
train_samples = []
valid_samples = {}
for qid in qrel:
    # key: doc id, value: relevance score
    dic_doc_id_relevance = qrel[qid]
    # query text
    topic_text = query_tokens[qid]

    if counter < number_training_samples:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            content = collection_dic[doc_id]
            if label >= 1:
                label = 1
            train_samples.append(InputExample(texts=[topic_text, content], label=label))
    else:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            if qid not in valid_samples:
                valid_samples[qid] = {'query': topic_text, 'positive': set(), 'negative': set()}
            if label == 0:
                label = 'negative'
            else:
                label = 'positive'
            content = collection_dic[doc_id]
            valid_samples[qid][label].add(content)
    counter += 1

print("Training and validation set prepared")

def fine_tune():

# selecting cross-encoder
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Learn how to use GPU with this!
model = CrossEncoder('ms-marco-MiniLM-L-6-v2')

# Adding special tokens
tokens = ["[TITLE]", "[BODY]"]
model.tokenizer.add_tokens(tokens, special_tokens=True)
model.model.resize_token_embeddings(len(model.tokenizer))

num_epochs = 2
model_save_path = "./ft_cr_2024"
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)
# During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CERerankingEvaluator(valid_samples, name='train-eval')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
train_loss = losses.MultipleNegativesRankingLoss(model=model)
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    save_best_model=True
)

model.save(model_save_path)
