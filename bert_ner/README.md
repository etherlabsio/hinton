## Requirements
 - transformers
 - CoNLL 2003 dataset renamed to train.txt, test.txt and dev.txt
## Training NER Model Script:
```
python run_ner.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --output_dir=/path/to/model-destination/bert-ner-uncased-9classes/ --data_dir=/path/to/data-dir/ --save_steps=20000 --num_train_epochs=3 --do_lower_case --evaluate_during_training --overwrite_cache --logging_steps=2000 --overwrite_output_dir
```
