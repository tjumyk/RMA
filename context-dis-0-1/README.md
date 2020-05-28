# Feature Generation
----

This project contians a series of python scripts only for the generation of the features. It fetches the documents from the database. And then it can extract some features from these raw documents. Finally, these features are written into the `Redis` databse for later use. 

### Project Structure 
----
0. `fea_gen_kp_redis.py` is the python script containing the `main` method. It provides 3 methods:
    1. `main`: main method, receive program parameters from the command line, including `experiment ID, ... `
    2. `main_multiprocessing`: it is used for generating the training cases or testing cases, and then process these cases, finally save these features to database.
    3. `save_res_2_redis`: used for writting the generated features to redis database.
1. Folder `const_vars` contians two scripts, `const_vars_abc.py` and `constant_conll_testb.py`. These scripts have some const variables that will be used during the whole process. We don't need to change it.
2. The folder `corpora` contains a python script `conll.py`, which is used to generate the training/tesing mentions and their priors from the database. Here, only 1 method is important, 
    1. `get_valid_mention_entity_pairs`. It accepts 4 parameters: `id_start, id_end`, the first and last mention ids. We will generate mentions between these two ids.`gen_all`, ignore it;  `data_type(train, testb, ...)`.
3. The folder `database` contains two python scripts, `db_kongzi2.py`: get the candidate entities and their prior; `db_kongzi3.py`: get the conll train/test cases. They are used in `conll.py`.
4.  The folder `feature_generator`, it contains the feature generation functions. Only  `fea_gen_token_phrase_idf_entropy.py` is used. It contains 3 important methods, among which `fea_gen_tf_idf_entropy_redis` is used to generate the `tf idf entropy` realted features.
5. The folder `redis_db` can be used to perform some operation on the redis database, such as saving the generated features.
6. The folder `tokenizer` cotains `spacy_tokenizer.py`, which is used to calculate the `tf idf entropy` for every token and entity. The calculation method is `main_gen_tf_idf_entropy_fea_redis`.

### How to run the script
----
#### From command line

This script needs 5 parameters, `start id, end id, data type, program name, experiment id`, the following is an example.
```
python fea_gen_kp_redis.py 371182 371183 train debug-a basic_fea_ctx
```