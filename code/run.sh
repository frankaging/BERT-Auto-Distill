# This is a template for training the first time
CUDA_VISIBLE_DEVICES=1,2,4 python run_auto_distill.py \
--task_name SST2 \
--data_dir ../sample_data/dataset/SST2/ \
--vocab_file ../models/uncased_L-12_H-768_A-12/vocab.txt \
--teacher_config_file ../models/uncased_L-12_H-768_A-12/bert_config.json \
--model_type TeacherBERT \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir ../results/SST2-TeacherBERT/ \
--seed 42 \
--init_checkpoint ../models/uncased_L-12_H-768_A-12/pytorch_model.bin