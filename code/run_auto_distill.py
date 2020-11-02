import argparse

from util.distill_helper import *
from torch.nn import CrossEntropyLoss

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# we include here as it is clearly as this is our main function
def step_distill(train_dataloader, test_dataloader, teacher_model, student_model,
                 optimizer, device, n_gpu, evaluate_interval, global_step, 
                 output_log_file, epoch, global_best_acc, args):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(pbar):
        teacher_model.eval()
        student_model.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # truncate to save space and computing resource
        input_ids, input_mask, segment_ids, label_ids, seq_lens = batch
        max_seq_lens = max(seq_lens)[0]
        input_ids = input_ids[:,:max_seq_lens]
        input_mask = input_mask[:,:max_seq_lens]
        segment_ids = segment_ids[:,:max_seq_lens]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        seq_lens = seq_lens.to(device)

        _, teacher_logits = \
            teacher_model(input_ids, segment_ids, input_mask, seq_lens,
                          device=device, labels=label_ids)

        student_loss, student_logits = \
            student_model(input_ids, segment_ids, input_mask, seq_lens,
                          device=device, labels=label_ids)

        if args.goal == "bd":
            # let us use simply student loss
            # (1) pred loss
            # (2) logit diff loss with techer model
            logit_loss_func = CrossEntropyLoss()
            logit_loss = logit_loss_func(student_logits, teacher_logits)
            student_loss += logit_loss
        elif args.goal == "rld":
            # TODO:
            # RL-based losses here!
            pass
        elif args.goal == "nd":
            pass
        elif args.goal == "pkd":
            # other baseline
            pass
        else:
            pass

        if n_gpu > 1:
            student_loss = student_loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            student_loss = student_loss / args.gradient_accumulation_steps
        student_loss.backward()
        tr_loss += student_loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()    # We have accumulated enought gradients
            student_model.zero_grad()
            global_step += 1
        pbar.set_postfix({'train_loss': student_loss.tolist()})

        if global_step % 500 == 0:
            logger.info("***** Evaluation Interval Hit *****")
            global_best_acc = evaluate(test_dataloader, student_model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
                                       global_step, output_log_file, global_best_acc, args)

    return global_step, global_best_acc

def main(args):

    device, n_gpu, output_log_file= system_setups(args)

    # data loader, we load the model and corresponding training and testing sets
    if args.model_type == "TeacherBERT":
        model, optimizer, train_dataloader, test_dataloader = \
            data_and_model_loader(device, n_gpu, args)
    elif args.model_type == "StudentBERT":
        teacher_model, student_model, optimizer, train_dataloader, test_dataloader = \
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
            # we are training a student model instead
            global_step, global_best_acc = \
                step_distill(train_dataloader, test_dataloader, 
                             teacher_model, student_model,
                             optimizer, device, n_gpu, evaluate_interval, global_step, 
                             output_log_file, epoch, global_best_acc, args)
        epoch += 1

    logger.info("***** Global best performance *****")
    logger.info("accuracy on dev set: " + str(global_best_acc))
    
if __name__ == "__main__":
    from util.args_parser import parser
    args = parser.parse_args()
    main(args)