---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:6211
- loss:CosineSimilarityLoss
base_model: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
widget:
- source_sentence: '[TITLE]Are countries Netherlands support hitchhiking[BODY]According
    Internet Netherlands support hitchhiking placing signs see picture spots good
    location hitchhikers Are countries great support hitchhikers Im asking easy countries
    hitchhike I want know governments make similar effort like installing signs legalize
    hitchhiking provide official information hitchhiking support hitchhikers'
  sentences:
  - Coin lockers feature stations Japan Depending size station lockers generally run
    200 300 JPY per day Lockers different sizes aside major rail stations e g Tokyo
    I think lockers large enough accommodate large roller bag like used checked luggage
    airlines
  - The problem many online comparison tools dont necessarily reflect kind travel
    lifestyle youre going Heres I normally get good estimate Go HostelWorld find highest
    rated hostel check daily rate 4 person room In Prague thats around 30 EUR London
    closer 50 EUR Go Numbeo check local restaurant prices Meal Inexpensive Restaurant
    section Go Uber check rates city countrys capital If taxi cheap enough Ill often
    completely skip public transport If taxi expensive Uber unavailable go Numbeo
    find price single public transport ticket Go Foursquare check entry fee top rated
    museum Surprisingly London actually cheap regard museums free almost free Since
    I dont spend much money outside categories I travel gives great idea exactly much
    I end spending
  - I done hitchhking Belgium Ardennes without much problems There official signs
    Btw I also didnt see Netherlands I live For netherlands I see regularly people
    hitchhiking good places stop gas stations especially along high ways get car first
    From go almost anywhere Make sign helps lot stations road network like spider
- source_sentence: '[TITLE]Hard case suitcase doesnt open side[BODY]I looking hard
    case light weight bag like one pictured however zipper top yellow line rather
    side This would allow open bag without using lot space original layout Has anybody
    seen bag like somebody know buy something like'
  sentences:
  - Do plan coming back US If dont sweat Texas isnt going ask country return US traffic
    citation Unless plan return ignore If plan return pay fine avoid hassles future
    return Bench warrants dont disappear future interactions police US likely result
    incarceration get old ticket warrant sorted Itll cost pretty penny As rudeness
    cop welcome US Cops aholes love pull guns shout people They get The thing like
    better shoot dog think get away A lot real sickos police country And wonder dont
    like
  - In Spain I purchased hard sided soft top bag zipper yellow line suggests But I
    bought chino store kinds things low cost mostly low quality Spanish call chino
    always run people look Chinese least one I went Vietnamese Cost 30 largest size
    bigger airlines check limit one cruise plane flight plastic made hard sided least
    dozen pieces part roller frame come apart I needed one trip expected worth three
    trips obviously I overestimated If made slightly better materials would actually
    good bag Unfortunately branding information single logo stuck front top fell way
    store But four sizes giant seat size
  - Since I arrived Japan April year I taking cold shower almost every day At first
    I I didnt know enable hot water I got used For easier exercising going run weight
    based exercises like pushups crunches cold shower feel like reward
- source_sentence: '[TITLE]Are walking tours join group Rotterdam[BODY]A weeks ago
    I Berlin huge number excellent walking tours many free pay tip end These covered
    usual sights well interesting less common things At home Oxford practically cant
    walk Broad Street without tripping group waiting one different mostly free walking
    tours start Im going Rotterdam weekend I hoping find something similar However
    I havent able find evidence tours I find details self guided ones I ideally want
    find one someone leading group around offering commentary pointing interesting
    things answering questions etc Do tours exist Rotterdam Or I spoilt Berlin Oxford
    thinking theyre common fact theyre perhaps bit exception'
  sentences:
  - Assuming EU citizen indeed allowed exit airport able return without problems long
    valid boarding pass next flight In fact Schiphol website suggests leave airport
    stroll around city four hours next flight The recommended check time latest two
    hours flying another European country least three flying outside Europe Schiphol
    one largest busiest airports Europe long waiting times 30 minutes security checks
    unusual
  - I know local customs handy piece equipment training barefoot walking running Vibram
    Five Finger shoes link google find easily If get size fits perfectly almost feel
    mechanically walk like barefoot minus pain rough surfaces Image attribution So
    eventually going distance barefoot spiritual goal means use training
  - There number great activities join Rotterdam If looking free tours best use Couchsurfing
    MeetUp find great local events Alternatively try getting touch hostels tend activities
    non guests sometimes A great one Ani Haakien I provide 4 Free Weekly Tours You
    find information directly Frank Tours Rotterdam I see someone already left comment
    Ani Haakien These tours additional Saturday Architecture Tour 11
- source_sentence: '[TITLE]Why prices published without tax US[BODY]On recent trip
    San Francisco always surprise see pay stores It never simple adding published
    prices There always addition taxes Sometimes additional 50ct sometimes increase
    price substantial The extreme case bag apples advertised price 1 99 final price
    4 50 Isnt single VAT I know price expect If taxes apply everyone simply publish
    price including taxes'
  sentences:
  - I would say law consumer side USA therefore require total price displayed Most
    shops therefore leave taxes etc likely buy item Trustworthily companies loose
    trade due companies misleading consumers prices therefore quickly companies become
    bad
  - Most likely scam Bank transfer equals scam Whatever website went unlikely actual
    owner property That kind scam costs people millions every year
  - There safe way high fence bars access long term parking lot To save money I would
    take Q10 bus terminal 5 transfer A train Lefferts Blvd F train Kew Gardens There
    buses B15 3 Train New Lots Q3 Locust Manor LIRR F Train Jamaica I familiar
- source_sentence: '[TITLE]Are man places world[BODY]I know forbidden enter Muslims
    mosques shrines without proper Islamic cover shoes walking temples tombs shrines
    India Hindu countries without proper cover going Christian church monastery I
    mean man place let woman kind cover religion proper considerations comes P S It
    divided places men women go separately man places'
  sentences:
  - But always preferred opinion depends airport security Someday Frankfort I brother
    tried attend together border control officer directly asked come one one So I
    think situation applies NY airport since security level high Not Sweden never
    ask
  - The Herbertstrae Herbert street Hamburg German It street prostitutes offering
    suitors sitting windows The street screens noone outside look street There red
    signs prohibit enter street minors women While former times minors women fact
    forbidden legally women able enter street public place It still strongly recommended
    womens enter prostitutes act extremely aggressive That means jeers taunts insults
    instances bucket fluids head I know rumor scare women away fact water insist prostitutes
    really used excrements
  - This policy exist previously mentioned It important understand reason policy exists
    delegate appropriate authority act individuals sworn duty protect countrys borders
    The policy intended give authority officers front lines border interacting travelers
    It allows situations things dont feel right Consider case incidents September
    11 2001 US Customs Border Patrol Officer Jose Melendez Perez received applause
    commissioners spectators described Aug 4 2001 refused allow Mohamed al Qahtani
    commission members described probable 20th Sept 11 hijacker enter Orlando airport
    based almost entirely gut feeling man lying I felt bone chilling cold effect said
    He gave chills Credit Baltimore Sun Sept 11 hijacker raised suspicions border
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
model-index:
- name: SentenceTransformer based on sentence-transformers/multi-qa-MiniLM-L6-cos-v1
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: pearson_cosine
      value: 0.19340408975209622
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.20352610798747794
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.1902683826432222
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.20172918641782703
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.19055393788160252
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.20352610798747794
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.1934040744290362
      name: Pearson Dot
    - type: spearman_dot
      value: 0.20352610798747794
      name: Spearman Dot
    - type: pearson_max
      value: 0.19340408975209622
      name: Pearson Max
    - type: spearman_max
      value: 0.20352610798747794
      name: Spearman Max
---

# SentenceTransformer based on sentence-transformers/multi-qa-MiniLM-L6-cos-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) <!-- at revision 2d981ed0b0b8591b038d472b10c38b96016aab2e -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("bi-ef-travel-qa-v1")
# Run inference
sentences = [
    '[TITLE]Are man places world[BODY]I know forbidden enter Muslims mosques shrines without proper Islamic cover shoes walking temples tombs shrines India Hindu countries without proper cover going Christian church monastery I mean man place let woman kind cover religion proper considerations comes P S It divided places men women go separately man places',
    'The Herbertstrae Herbert street Hamburg German It street prostitutes offering suitors sitting windows The street screens noone outside look street There red signs prohibit enter street minors women While former times minors women fact forbidden legally women able enter street public place It still strongly recommended womens enter prostitutes act extremely aggressive That means jeers taunts insults instances bucket fluids head I know rumor scare women away fact water insist prostitutes really used excrements',
    'This policy exist previously mentioned It important understand reason policy exists delegate appropriate authority act individuals sworn duty protect countrys borders The policy intended give authority officers front lines border interacting travelers It allows situations things dont feel right Consider case incidents September 11 2001 US Customs Border Patrol Officer Jose Melendez Perez received applause commissioners spectators described Aug 4 2001 refused allow Mohamed al Qahtani commission members described probable 20th Sept 11 hijacker enter Orlando airport based almost entirely gut feeling man lying I felt bone chilling cold effect said He gave chills Credit Baltimore Sun Sept 11 hijacker raised suspicions border',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.1934     |
| spearman_cosine    | 0.2035     |
| pearson_manhattan  | 0.1903     |
| spearman_manhattan | 0.2017     |
| pearson_euclidean  | 0.1906     |
| spearman_euclidean | 0.2035     |
| pearson_dot        | 0.1934     |
| spearman_dot       | 0.2035     |
| pearson_max        | 0.1934     |
| **spearman_max**   | **0.2035** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 6,211 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | float                                                          |
  | details | <ul><li>min: 23 tokens</li><li>mean: 94.16 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 105.31 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.56</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>[TITLE]Is possible travel cargo airplanes[BODY]Some time ago I heard someone telling travelling cargo planes Apparently planes limited amount extra seats companies would sell extra profit Does anyone information Is still possible</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>Short answer I special knowledge subject propose google potpourri All I know I know nothing except Google Socrates It 2012 aprils fool STA travel Apparently possible Fedex employees point Fly cargo plane Passengers cargo flights still possible If pet possible pet cargo Maybe case special merchandise professionals stay near travel 3 zoo keepers flew cargo brought two pandas By boat feasible I get information freighter travel</code>                                                                                                                                                | <code>0.0</code> |
  | <code>[TITLE]Is tipping mandatory Restaurants Bars Germany[BODY]In Bavaria I paid drink I asked Ruckgeld got integer part money back rest kept tip I looked eye understood I wasnt happy initiative wouldnt blink eye My company advised let go mentioning I going ask change would swear Notice didnt return correct amount back stare demanding tip would decide Since I around week I thinking paying exact amount money next time Will I get uncomfortable situation waiter stare demanding tip In words tipping mandatory Germany PS Having experienced tipping cruelty USA tipped generously waiters across Mediterranean supreme behavior Saturday night came shock since I impression awful habit USA thing know like fake audience laughing USA TV series</code> | <code>The waitress substantially rude I dont understand company advising let non German The correct answer would x cents missing German Da fehlen x cent mildly angry tone voice While tipping common restaurants Germany mandatory always seen something nice voluntarily Waitors may give ample opportunity decide tip e g slowly searching coins purse might well shuffling around barring statement customer exact change given ask If I witness behaviour even directed I would remove whatever I intended tip bill</code>                                                                         | <code>0.0</code> |
  | <code>[TITLE]Where I eat medieval food[BODY]Medieval cuisine roughly 5th 15th century Europe plethora dishes basically unknown modern palate frumenty pottage sops possets etc Are restaurants I eat today Primarily interested medieval European cuisine open answers elsewhere world Honorary mention Cardo Culinaria Jerusalem tried recreate authentic ancient Roman dining experience It abandon pretense authenticity years though persnickety customers insisting forks instead eating hand etc eventually closed sometime early 2000s Update Ive accepted best answer date still open since actual restaurant listed far quite fits bill</code>                                                                                                                   | <code>You could try look experimental archeology groups see find one interested cooking middle ages near Experimental archeology way test hypotheses actually physically trying And done scientists hobby people well In Netherlands living history museum recreated dwellings crafts prehistory middle ages I dont remember medieval food served herbal tea medieval village served earthen mug So maybe food times A similar thing Viking weekends enacted volunteers history nerds It depend specific individuals dying fabrics viking way agriculture leartherworking You find group foodies</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 100
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 100
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch   | Step  | Training Loss | spearman_max |
|:-------:|:-----:|:-------------:|:------------:|
| 1.0     | 389   | -             | 0.0913       |
| 1.2853  | 500   | 0.2478        | -            |
| 2.0     | 778   | -             | 0.1493       |
| 2.5707  | 1000  | 0.2369        | -            |
| 3.0     | 1167  | -             | 0.1884       |
| 3.8560  | 1500  | 0.2235        | -            |
| 4.0     | 1556  | -             | 0.2056       |
| 5.0     | 1945  | -             | 0.2053       |
| 5.1414  | 2000  | 0.2126        | -            |
| 6.0     | 2334  | -             | 0.1886       |
| 6.4267  | 2500  | 0.1966        | -            |
| 7.0     | 2723  | -             | 0.1666       |
| 7.7121  | 3000  | 0.1744        | -            |
| 8.0     | 3112  | -             | 0.1667       |
| 8.9974  | 3500  | 0.1473        | -            |
| 9.0     | 3501  | -             | 0.1365       |
| 10.0    | 3890  | -             | 0.1630       |
| 10.2828 | 4000  | 0.1198        | -            |
| 11.0    | 4279  | -             | 0.1780       |
| 11.5681 | 4500  | 0.0925        | -            |
| 12.0    | 4668  | -             | 0.1526       |
| 12.8535 | 5000  | 0.0719        | -            |
| 13.0    | 5057  | -             | 0.1529       |
| 14.0    | 5446  | -             | 0.1049       |
| 14.1388 | 5500  | 0.055         | -            |
| 15.0    | 5835  | -             | 0.1375       |
| 15.4242 | 6000  | 0.0407        | -            |
| 16.0    | 6224  | -             | 0.1087       |
| 16.7095 | 6500  | 0.0321        | -            |
| 17.0    | 6613  | -             | 0.1201       |
| 17.9949 | 7000  | 0.0284        | -            |
| 18.0    | 7002  | -             | 0.1311       |
| 19.0    | 7391  | -             | 0.1302       |
| 19.2802 | 7500  | 0.0243        | -            |
| 20.0    | 7780  | -             | 0.1055       |
| 20.5656 | 8000  | 0.0194        | -            |
| 21.0    | 8169  | -             | 0.1638       |
| 21.8509 | 8500  | 0.017         | -            |
| 22.0    | 8558  | -             | 0.1614       |
| 23.0    | 8947  | -             | 0.1486       |
| 23.1362 | 9000  | 0.015         | -            |
| 24.0    | 9336  | -             | 0.1501       |
| 24.4216 | 9500  | 0.0153        | -            |
| 25.0    | 9725  | -             | 0.1450       |
| 25.7069 | 10000 | 0.0124        | -            |
| 26.0    | 10114 | -             | 0.1556       |
| 26.9923 | 10500 | 0.0117        | -            |
| 27.0    | 10503 | -             | 0.0947       |
| 28.0    | 10892 | -             | 0.0833       |
| 28.2776 | 11000 | 0.01          | -            |
| 29.0    | 11281 | -             | 0.1056       |
| 29.5630 | 11500 | 0.0087        | -            |
| 30.0    | 11670 | -             | 0.1589       |
| 30.8483 | 12000 | 0.008         | -            |
| 31.0    | 12059 | -             | 0.1393       |
| 32.0    | 12448 | -             | 0.1682       |
| 32.1337 | 12500 | 0.0069        | -            |
| 33.0    | 12837 | -             | 0.1762       |
| 33.4190 | 13000 | 0.0064        | -            |
| 34.0    | 13226 | -             | 0.2286       |
| 34.7044 | 13500 | 0.0061        | -            |
| 35.0    | 13615 | -             | 0.1588       |
| 35.9897 | 14000 | 0.0058        | -            |
| 36.0    | 14004 | -             | 0.1531       |
| 37.0    | 14393 | -             | 0.1677       |
| 37.2751 | 14500 | 0.0052        | -            |
| 38.0    | 14782 | -             | 0.1357       |
| 38.5604 | 15000 | 0.0055        | -            |
| 39.0    | 15171 | -             | 0.1829       |
| 39.8458 | 15500 | 0.0039        | -            |
| 40.0    | 15560 | -             | 0.0856       |
| 41.0    | 15949 | -             | 0.1132       |
| 41.1311 | 16000 | 0.0044        | -            |
| 42.0    | 16338 | -             | 0.2259       |
| 42.4165 | 16500 | 0.0046        | -            |
| 43.0    | 16727 | -             | 0.1940       |
| 43.7018 | 17000 | 0.0042        | -            |
| 44.0    | 17116 | -             | 0.1980       |
| 44.9871 | 17500 | 0.0029        | -            |
| 45.0    | 17505 | -             | 0.1842       |
| 46.0    | 17894 | -             | 0.1893       |
| 46.2725 | 18000 | 0.0038        | -            |
| 47.0    | 18283 | -             | 0.2319       |
| 47.5578 | 18500 | 0.0031        | -            |
| 48.0    | 18672 | -             | 0.1798       |
| 48.8432 | 19000 | 0.0032        | -            |
| 49.0    | 19061 | -             | 0.1438       |
| 50.0    | 19450 | -             | 0.2038       |
| 50.1285 | 19500 | 0.0022        | -            |
| 51.0    | 19839 | -             | 0.2073       |
| 51.4139 | 20000 | 0.0032        | -            |
| 52.0    | 20228 | -             | 0.1789       |
| 52.6992 | 20500 | 0.0024        | -            |
| 53.0    | 20617 | -             | 0.1734       |
| 53.9846 | 21000 | 0.002         | -            |
| 54.0    | 21006 | -             | 0.2172       |
| 55.0    | 21395 | -             | 0.1937       |
| 55.2699 | 21500 | 0.0023        | -            |
| 56.0    | 21784 | -             | 0.2106       |
| 56.5553 | 22000 | 0.0022        | -            |
| 57.0    | 22173 | -             | 0.1935       |
| 57.8406 | 22500 | 0.0015        | -            |
| 58.0    | 22562 | -             | 0.1871       |
| 59.0    | 22951 | -             | 0.1996       |
| 59.1260 | 23000 | 0.0018        | -            |
| 60.0    | 23340 | -             | 0.2049       |
| 60.4113 | 23500 | 0.0013        | -            |
| 61.0    | 23729 | -             | 0.1715       |
| 61.6967 | 24000 | 0.0015        | -            |
| 62.0    | 24118 | -             | 0.2038       |
| 62.9820 | 24500 | 0.0018        | -            |
| 63.0    | 24507 | -             | 0.2121       |
| 64.0    | 24896 | -             | 0.2008       |
| 64.2674 | 25000 | 0.0013        | -            |
| 65.0    | 25285 | -             | 0.2108       |
| 65.5527 | 25500 | 0.002         | -            |
| 66.0    | 25674 | -             | 0.1903       |
| 66.8380 | 26000 | 0.0008        | -            |
| 67.0    | 26063 | -             | 0.1930       |
| 68.0    | 26452 | -             | 0.1743       |
| 68.1234 | 26500 | 0.0013        | -            |
| 69.0    | 26841 | -             | 0.2008       |
| 69.4087 | 27000 | 0.0011        | -            |
| 70.0    | 27230 | -             | 0.1851       |
| 70.6941 | 27500 | 0.0007        | -            |
| 71.0    | 27619 | -             | 0.2200       |
| 71.9794 | 28000 | 0.0014        | -            |
| 72.0    | 28008 | -             | 0.1924       |
| 73.0    | 28397 | -             | 0.2183       |
| 73.2648 | 28500 | 0.001         | -            |
| 74.0    | 28786 | -             | 0.1862       |
| 74.5501 | 29000 | 0.0009        | -            |
| 75.0    | 29175 | -             | 0.1998       |
| 75.8355 | 29500 | 0.0009        | -            |
| 76.0    | 29564 | -             | 0.2212       |
| 77.0    | 29953 | -             | 0.2214       |
| 77.1208 | 30000 | 0.001         | -            |
| 78.0    | 30342 | -             | 0.1931       |
| 78.4062 | 30500 | 0.0007        | -            |
| 79.0    | 30731 | -             | 0.2378       |
| 79.6915 | 31000 | 0.0005        | -            |
| 80.0    | 31120 | -             | 0.2284       |
| 80.9769 | 31500 | 0.0008        | -            |
| 81.0    | 31509 | -             | 0.2203       |
| 82.0    | 31898 | -             | 0.2166       |
| 82.2622 | 32000 | 0.0007        | -            |
| 83.0    | 32287 | -             | 0.2084       |
| 83.5476 | 32500 | 0.0006        | -            |
| 84.0    | 32676 | -             | 0.1800       |
| 84.8329 | 33000 | 0.0003        | -            |
| 85.0    | 33065 | -             | 0.2105       |
| 86.0    | 33454 | -             | 0.1871       |
| 86.1183 | 33500 | 0.0005        | -            |
| 87.0    | 33843 | -             | 0.2159       |
| 87.4036 | 34000 | 0.0003        | -            |
| 88.0    | 34232 | -             | 0.2074       |
| 88.6889 | 34500 | 0.0003        | -            |
| 89.0    | 34621 | -             | 0.2085       |
| 89.9743 | 35000 | 0.0004        | -            |
| 90.0    | 35010 | -             | 0.2046       |
| 91.0    | 35399 | -             | 0.2107       |
| 91.2596 | 35500 | 0.0001        | -            |
| 92.0    | 35788 | -             | 0.2093       |
| 92.5450 | 36000 | 0.0002        | -            |
| 93.0    | 36177 | -             | 0.2172       |
| 93.8303 | 36500 | 0.0002        | -            |
| 94.0    | 36566 | -             | 0.2211       |
| 95.0    | 36955 | -             | 0.2182       |
| 95.1157 | 37000 | 0.0003        | -            |
| 96.0    | 37344 | -             | 0.1998       |
| 96.4010 | 37500 | 0.0002        | -            |
| 97.0    | 37733 | -             | 0.1987       |
| 97.6864 | 38000 | 0.0001        | -            |
| 98.0    | 38122 | -             | 0.2028       |
| 98.9717 | 38500 | 0.0001        | -            |
| 99.0    | 38511 | -             | 0.2029       |
| 100.0   | 38900 | -             | 0.2035       |

</details>

### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.1.1
- Transformers: 4.45.2
- PyTorch: 2.5.1+cu124
- Accelerate: 1.1.0
- Datasets: 3.1.0
- Tokenizers: 0.20.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->