# project_nlp_chen
nlp-roth, lmu 2019


This repository only contains the codes for the project, the data is saved in `/big/b/beroth/abgaben/chenyi` in CIP-pool.


### the structure of folders of the whole project:

```
|-data
  |-word2vec_train_untyped_unique_filtered.txt
    (deduplicated and contain only paths/entities which occur at least once in test_simplified.csv)
|-data_  
    (the data used in this project, negative factor 5, contain entities which occur at least twice in test_simplified.csv)
  |-ent2id.json
  |-entities_path_filtered.txt
  |-entpair2id.json
  |-path2id.json
  |-path_pos_neg.txt (dobj^-___join___nsubj m.0_00#m.01gf5z m.05cgv#m.0d05l6 : path positive_entity_pair negative_entity_pair
|-experiments (only contain the best results in each model)
|-models
  |-build_data.py  (generate data for those models which only use entity_pair embeddings)
  |-build_data_.py (generate data for those models which use both entity_pair/entity embeddings)
  |-plot_loss.py
  |-train_loop.py (training loop for models)
  |-seq_model.py (model use sequential layers, sigmoid(pathT pos) ~1, sigmoid(pathTneg)~0, only use entity_pair embeddings)
  |-uni_model.py (universal schema, ranked pairs, sigmoid(pathTpos - pathTneg)~1, only use entity_pair embeddings)
  |-uni_model.py (universal schema, use both entity_pair/entity embeddings)
  |-word2vec.py (use word2vec model, sigmoid(pathTpos)~1, sigmoid(pathTneg)~0, only use entity_pair embeddings)
  |-word2vec_ent.py(word2vec model, use both entity_pair/entity embeddings)
evaluate.py (evaluate the result, call: python evaluate.py experiments/XXmodel/XX_embeddings.csv test_simplified.csv)
load_model.py (load the saved model state_dict and output the embeddings)
parameter_search.py 
test_simplified.csv ( for test)
train_seq_model.py (train model from models/seq_model.py)
train_uni_model.py (train model from models/uni_model.py)
train_uni_model_ent.py (train model from models/uni_model_ent.py)
train_word2vec.py (train model from models/word2vec.py)
train_word2vec_ent.py (train model from models/word2vec_ent.py)
```

### dataset
1. deduplicated the given data `word2vec_train_untyped.txt`, and filtered the data so that it only contains the paths, entities which has the paths that occurs at least once in `test_simplified.csv` (given data) also `untypify` the `paths` using `filter_path`, to get `word2vec_train_untyped_unique_filtered.txt`.

2. run `processing/main.py`to 

2.1. get the frequency of entities `ent_freq.txt`, the frequency of path `path_freq.txt`

2.2. use the frequency to filter out the data which contains the entities that occur at least twice to get the data which is used for generating train data and dev data `entities_path_filtered.txt`, `entpair2id.json`, `ent2id.json`, `path2id.json`.

3. run `preprocessing/rank_pairs.py` to do negative sampling on each `path positive_entity_pair` , negative sampling factor is 5. That means, for each `path positive_entity_pair`, there are five different `negative_entity_pair` which has not occur in `positive_entity_pair` corresponding to the `path`, but the entity pairs already occur in the dataset ==> get the data `data_/path_pos_neg.txt` for generating the train/dev dataset.

* dataset :
```
train_data samples: 2031332
dev_data samples: 20518 (1% of total samples)
total number of samples: 2051850
number of paths: 1714 (1717 in test_simplified.csv)
number of entity pairs: 66835
number of entities: 19396 
```

### results:
the learned word embeddings are in `experiments/xxmodel/xx_embeddings.csv`

only present the best results from each experimented model
```text
baseline `python evaluate relation_embeddings.txt test_simplified.csv`:
Results based on corss validation on 3985 samples with 5 folds.
Mean Precision: 0.471
Mean Recall: 0.712
Mean F1: 0.565

Sequential Model
python evaluate.py experiments/seqmodel/1_lstm_linear_adam_0.01_20_4096_model.pt_embeddings.csv test_simplified.csv
Loading embeddings ... Done.
Results based on cross validation on 3983 samples with 5 folds.
Mean Precision: 0.418
Mean Recall: 0.809
Mean F1: 0.551

UniModel (universal schema, ranked pairs, 20 epochs, 4096 batch size, Adam lr = 0.01)
Results based on cross validation on 3983 samples with 5 folds.
Mean Precision: 0.549
Mean Recall: 0.787
Mean F1: 0.646

Word2Vec Model (20 epochs, 4096 batch size, adam lr=0.01)
Results based on cross validation on 3983 samples with 5 folds.
Mean Precision: 0.566
Mean Recall: 0.720
Mean F1: 0.633

Universal Schema using both entity_pair and entity embeddings (30 epochs, Adam lr=0.01, batchsize 4096)
 python evaluate.py experiments/unimodel_ent/adam_0.01_amsgrad_30_4096_model.pt_embeddings.csv test_simplified.csv
Loading embeddings ... Done.
Results based on cross validation on 3983 samples with 5 folds.
Mean Precision: 0.507
Mean Recall: 0.796
Mean F1: 0.618


Word2Vec Model using both entity_pair and entity embeddings (40 epochs, Adam 0.01, batch size 4096)
python evaluate.py experiments/word2vec_ent/adam_0.01_40_4096_model.pt_embeddings_01.csv test_simplified.csv
Loading embeddings ... Done.
Results based on cross validation on 3983 samples with 5 folds.
Mean Precision: 0.518
Mean Recall: 0.777
Mean F1: 0.621

```

### remarks:
* use data structure as `path pos_entity_pair neg_entity_pair` is much less memory consuming than `path pos_entity_pair True` and `path neg_entity_pair False`
* to report the dev_loss and train_loss, detach the tensor and only record the number `loss_dict["dev"].append(dev_loss.detach().item())` to avoid `out of memory` error
* to concatenate the embeddings of `entity1` and `entity2` in order to get the embeddings of `entity1#entity2` does not work well for training the models, better to use embeddings of `entity1#entity2` directly or embeddings of `entity1#entity2` plus embeddings of `entity1` and embeddings of `entity2`.
* the model with neural networks does not work better than word2vec and unviersal schema, maybe due to the very short sequence length 2. (`path pos_entity_pair` or `path neg_entity_pair` as one sequence)
