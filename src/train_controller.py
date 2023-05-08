import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import cma
from controller import CONTROLLER
from tqdm import tqdm
from rollout_generator import RolloutGenerator, load_parameters, flatten_parameters
from dataclasses import dataclass, asdict


import numpy as np
import random
import torch
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
# https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html
@dataclass
class HParams():
    
    discrete_action: bool = True
    num_actions: int = 4
    latent_size: int = 32
    hidden_size: int = 256
    action_dim: int = 1
    n_workers: int= 4
    pop_size:  int=100
    target_return: int = 0.3
    n_samples: int = 5
    sigma: int = 100
    n_epochs: int = 10000
hparams = asdict(HParams())

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index,tmp_dir,hparams):
    """ Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(hparams, device)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(solutions, results, p_queue, r_queue, rollouts=200):
    """ Give current controller evaluation.
    Evaluation is minus the cumulated reward averaged over rollout runs.
    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts
    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)

def train(hparams):
    
    num_workers = hparams["n_workers"]
    pop_size = hparams["pop_size"]
    target_return = hparams["target_return"]
    n_samples = hparams["n_samples"]

    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()
    
    log_dir = ".."
    tmp_dir = join(log_dir, 'tmp')
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))
    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index,tmp_dir, hparams)).start()
    controller = CONTROLLER(hparams)  # dummy instance
    ctrl_file = "../controller.ckpt"
    # define current best and load parameters
    cur_best = None
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        # take params , mapping on cpu
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'}) 
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))
    parameters = controller.parameters()
 
    sigma = hparams["sigma"]
    es = cma.evolution_strategy.CMAEvolutionStrategy(flatten_parameters(parameters), sigma, {'popsize': pop_size})
    epoch = 0
    log_step = 3
    max_epoch = hparams["n_epochs"]
    while not es.stop() and epoch < max_epoch:
        if cur_best is not None and - cur_best > target_return:
            print("Already better than target, breaking...")
            break
        r_list = [0] * pop_size  # result list
        solutions = es.ask_geno()

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(0.01)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples

        es.tell(solutions, r_list)
        es.logger.add()
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list, p_queue, r_queue)
            print("Current evaluation: {} vs current best {}".format(best, cur_best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                    'reward': - cur_best,
                    'state_dict': controller.state_dict()},
                    ctrl_file)
            if - best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break
        epoch += 1
    e_queue.put('EOP')
    es.result_pretty()
    #sav
    es.logger.plot(fontsize=3)
    print("Saving results...")
    cma.s.figsave(log_dir+'/result_of_controller_train.svg')

if __name__ == "__main__":
    train(hparams)