import argparse

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--task_name",
                    default=None,
                    type=str,
                    required=True,
                    choices=["SST2"],
                    help="The name of the task to train.")
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--vocab_file",
                    default=None,
                    type=str,
                    required=True,
                    help="The vocabulary file that the model was trained on.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument('--model_type', 
                    type=str,
                    default=None,
                    required=True,
                    choices=["TeacherBERT", "StudentBERT"],
                    help='type of model to train')
# optional arguments
parser.add_argument('--alg', 
                    type=str,
                    default="bd",
                    choices=["bd", "nd", 'rld', 'pkd', 'kd'],
                    help='what type of distillation we are looking for? \n'
                         'bd (baseline distill), nd (no distill), rld (rl-based distill), '
                         'pkd and kd (other type of sota distill algs)')
parser.add_argument('--teacher_model_path', 
                    type=str,
                    default=None,
                    help='the path to teacher model if we are distilling')   
parser.add_argument("--init_checkpoint",
                    default=None,
                    type=str,
                    help="Initial checkpoint (usually from a pre-trained model).")
parser.add_argument("--teacher_config_file",
                    default=None,
                    type=str,
                    help="The config json file corresponding to the teacher BERT model.")   
parser.add_argument("--student_config_file",
                    default=None,
                    type=str,
                    help="The config json file corresponding to the student BERT model.")   
# some default argments we don't usually change
parser.add_argument("--eval_test",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the test set.")                    
parser.add_argument("--do_lower_case",
                    default=False,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--train_batch_size",
                    default=20,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=20,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--base_learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--accumulate_gradients",
                    type=int,
                    default=1,
                    help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', 
                    type=int, 
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")   