# Training teacher model
CUDA_VISIBLE_DEVICES=5,7 python run_auto_distill.py \
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
--num_train_epochs 3 \
--output_dir ../results/SST2-TeacherBERT/ \
--seed 42 \
--init_checkpoint ../models/uncased_L-12_H-768_A-12/pytorch_model.bin

# Training/distilling student model
CUDA_VISIBLE_DEVICES=5,7 python run_auto_distill.py \
--task_name SST2 \
--data_dir ../sample_data/dataset/SST2/ \
--vocab_file ../models/student_model/vocab.txt \
--student_config_file ../models/student_model/bert_config.json \
--model_type StudentBERT \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ../results/SST2-StudentBERT/ \
--seed 42 \
--teacher_config_file ../models/uncased_L-12_H-768_A-12/bert_config.json \
--teacher_model_path ../models/student_model/best_checkpoint.bin \
--alg bd