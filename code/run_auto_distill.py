import argparse

from util.distill_helper import *

def main(args):

    device, n_gpu, output_log_file= system_setups(args)

    # data loader, we load the model and corresponding training and testing sets
    model, optimizer, train_dataloader, test_dataloader = \
        data_and_model_loader(device, n_gpu, args)

    # main training step    
    global_step = 0
    global_best_acc = -1
    epoch=0
    evaluate_interval = 500
    # training epoch to eval
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        if args.model_type == "TeacherBERT":
            # train a teacher model solving this task
            global_step, global_best_acc = \
                step_train(train_dataloader, test_dataloader, model, optimizer, 
                           device, n_gpu, evaluate_interval, global_step, 
                           output_log_file, epoch, global_best_acc, args)
        elif args.model_type == "StudentBERT":
            pass
        epoch += 1

if __name__ == "__main__":
    from util.args_parser import parser
    args = parser.parse_args()
    main(args)