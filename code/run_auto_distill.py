import argparse

from util.distill_helper import *
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid

import torch.optim as optim
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def imitation_distance(teacher_states, student_states, seq_mask, dist_type="pkd"):
    """
    this part, we can take advantage of previous papers.
    Eqn. (7) for this paper https://arxiv.org/pdf/1908.09355.pdf
    is implemented here.
    """
    if isinstance(teacher_states, list):
        teacher_states = torch.stack(teacher_states, dim=0)
    if isinstance(student_states, list):
        student_states = torch.stack(student_states, dim=0)
    
    if dist_type == "pkd":
        # taking the unit value
        teacher_states_d = teacher_states.detach().data # detach it
        student_states_d = student_states.detach().data

        teacher_states_n = teacher_states_d.norm(p=2, dim=-1, keepdim=True)
        student_states_n = student_states_d.norm(p=2, dim=-1, keepdim=True)

        teacher_states_normalized = teacher_states_d.div(teacher_states_n)
        student_states_normalized = student_states_d.div(student_states_n)

        pkd_dist = teacher_states_normalized - student_states_normalized
        pkd_dist = pkd_dist.norm(p=2, dim=-1, keepdim=True).pow(2)
        pkd_dist = pkd_dist.sum(dim=0).sum(dim=-1)
        # masking
        pkd_dist = pkd_dist * seq_mask
        return pkd_dist.mean() # using the same reduction

# we include here as it is clearly as this is our main function
def step_distill(train_dataloader, test_dataloader, teacher_model, student_model,
                 rl_agents, rl_env, optimizerA, optimizerC, optimizer, device, n_gpu, evaluate_interval, global_step, 
                 output_log_file, epoch, global_best_acc, args):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(train_dataloader, desc="Iteration")

    # RL configs
    rl_coldstart= True
    prev_action, prev_dist, prev_value = None, None, None
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

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

        _, teacher_logits, teacher_env_encoder, _ = \
            teacher_model(input_ids, segment_ids, input_mask, seq_lens,
                          device=device, labels=label_ids)

        student_loss, student_logits, student_env_encoder, _ = \
            student_model(input_ids, segment_ids, input_mask, seq_lens,
                          device=device, labels=label_ids)

        if args.alg == "bd":
            # let us use simply student loss
            # (1) pred loss
            # (2) logit diff loss with techer model
            logit_loss_func = BCELoss()
            logits_to_prob = Sigmoid()
            logit_loss = logit_loss_func(logits_to_prob(student_logits), 
                                         logits_to_prob(teacher_logits.detach().data))
            student_loss += logit_loss

        elif args.alg == "rld":
            # adding bd loss
            logit_loss_func = BCELoss()
            logits_to_prob = Sigmoid()
            logit_loss = logit_loss_func(logits_to_prob(student_logits), 
                                         logits_to_prob(teacher_logits.detach().data))
            #####
            # i think we should calculate RL reward here.
            # for the first step, RL is not effecting the result
            # thus reward is delayed till a start
            # this reward is for the previous RL actions.
            if rl_coldstart:
                rl_coldstart = False
            else:
                reward = -1.0 * student_loss
                log_prob = prev_dist.log_prob(prev_action).unsqueeze(0)
                entropy += prev_dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(prev_value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            #####

            # get rl agents
            (actor, critic) = rl_agents
            # get env
            t_env = teacher_env_encoder[-1] # last layer encoder output
            s_env = student_env_encoder[-1]
            rl_env.update(teacher_env_encoder) # the env is all teacher's model internal states
            # form state vector, we only want a single decison per sentence
            input_state = torch.cat([t_env, s_env], dim=-1) # [b, seq, dim*2]
            input_state = input_state[:,0] # using CLS token
            # evaluate
            dist, value = actor(input_state), critic(input_state)
            # sample action
            imitate_count = student_model.bert.config.num_hidden_layers
            action = dist.sample(sample_shape=(imitate_count, ))
            # take action
            imitation_states = rl_env.step(action) # the imitation_states should 
                                                    # contains a entry with the same
                                                    # shape as student_env_encoder
            imi_dist = \
                    imitation_distance(imitation_states, 
                                       student_env_encoder, 
                                       input_mask) # distance is not reward.
                                                   # RL is learning to pick the
                                                   # right layer, not the similiar
                                                   # layer. RL is rewarded based 
                                                   # on hindsight loss.
            # the reward is only calculatable once the model is evaluated
            # using the updated weights.
            prev_action =  action
            prev_dist = dist
            prev_value = value

            # dnn loss
            student_loss += logit_loss
            student_loss += imi_dist

        elif args.alg == "nd":
            pass
        elif args.alg == "pkd":
            # other baseline
            pass
        elif args.alg == "rrld":
            # let us try a random RL agent here
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

        # RL learning
        # log_probs = torch.cat(log_probs)
        # returns = torch.cat(returns).detach()
        # values = torch.cat(values)

        # advantage = returns - values

        # actor_loss = -(log_probs * advantage.detach()).mean()
        # critic_loss = advantage.pow(2).mean()

        # optimizerA.zero_grad()
        # optimizerC.zero_grad()
        # actor_loss.backward()
        # critic_loss.backward()
        # optimizerA.step()
        # optimizerC.step()

        # dnn evaluation
        if global_step % 500 == 0:
            logger.info("***** Evaluation Interval Hit *****")
            global_best_acc = evaluate(test_dataloader, student_model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
                                       global_step, output_log_file, global_best_acc, args)

    return global_step, global_best_acc

def main(args):

    device, n_gpu, output_log_file = system_setups(args)

    # data loader, we load the model and corresponding training and testing sets
    if args.model_type == "TeacherBERT":
        model, optimizer, train_dataloader, test_dataloader = \
            data_and_model_loader(device, n_gpu, args)
    elif args.model_type == "StudentBERT":
        teacher_model, student_model, rl_agents, rl_env, optimizer, \
            train_dataloader, test_dataloader = \
            data_and_model_loader(device, n_gpu, args)
        # TODO: add a argument about it
        if False:
            # we will first evaluate teacher model as a target accuracy
            logger.info("***** Evaluation Teacher Model *****")
            _ = evaluate_fast(test_dataloader, teacher_model, device, n_gpu, args)

    # reset the rl env
    rl_env.reset()
    optimizerA = optim.Adam(rl_agents[0].parameters())
    optimizerC = optim.Adam(rl_agents[1].parameters())
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
                             teacher_model, student_model, rl_agents, rl_env, optimizerA, optimizerC, 
                             optimizer, device, n_gpu, evaluate_interval, global_step, 
                             output_log_file, epoch, global_best_acc, args)
        epoch += 1

    logger.info("***** Global best performance *****")
    logger.info("accuracy on dev set: " + str(global_best_acc))
    
    # saving RL agents
    # torch.save(actor, 'model/actor.pkl')
    # torch.save(critic, 'model/critic.pkl')

if __name__ == "__main__":
    from util.args_parser import parser
    args = parser.parse_args()
    main(args)