import pprint
import torch
import os


class Parameters:
    def __init__(self, cla, init=True):
        if not init:
            self.ls1 = 400
            self.ls2 = 300
            return
        cla = cla.parse_args()

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device('cuda')
        else:
            print("Using CPU")
            self.device = torch.device('cpu')

        # Render episodes
        self.render = cla.render
        self.env_name = cla.env
        self.save_periodic = cla.save_periodic

        # Number of Frames to Run
        if cla.env == 'Hopper-v2':
            self.num_frames = 4000000
        elif cla.env == 'Ant-v2' or cla.env == 'Walker2d-v2' or cla.env == 'HalfCheetah-v2':
            self.num_frames = 6000000
        else:   
            self.num_frames = 2000000

        if cla.env == 'Ant-v2' or cla.env == 'HalfCheetah-v2':
            self.warm_up = 10000
        else:   
            self.warm_up = 5000

        # Model save frequency if save is active
        self.next_save = cla.next_save
        self.next_frame_save = cla.next_frame_save

        # DARC params
        self.use_ln = True
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.q_weight = 0.1
        self.regularization_weight = 0.005
        self.seed = cla.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls1 = 400
        self.ls2 = 300

        # ========================================== NeuroEvolution Params =============================================
        # Num of trials
        self.num_evals = 1
        self.num_test_evals = cla.num_test_evals

        # Elitism Rate
        if cla.env == 'Reacher-v2' or cla.env == 'Walker2d-v2' or cla.env == 'Ant-v2' or cla.env == 'Hopper-v2':
            self.elite_fraction = 0.2
        else:
            self.elite_fraction = 0.1

        # Stable elite mechanism
        self.stable = cla.stable

        # Number of actors in the population
        self.pop_size = 10

        # Mutation and crossover
        self.mutation_prob = 0.7
        self.mutation_mag = cla.mut_mag
        self.mutation_noise = cla.mut_noise
        self.mutation_batch_size = 256
        self.mutation_epoches = 500
        self.proximal_mut = cla.proximal_mut
        self.phenotypic_mut = cla.phenotypic_mut
        self.distil = cla.distil
        self.distil_type = cla.distil_type
        self.verbose_mut = cla.verbose_mut
        self.verbose_crossover = cla.verbose_crossover
        self.verbose_rl_eval = cla.verbose_rl_eval

        # Mutation scale for Differential Mutation
        if cla.env == 'HalfCheetah-v2':
            self.mutation_scale = 0.3
        elif cla.env == 'Walker2d-v2':
            self.mutation_scale = 0.5
        elif cla.env == 'Hopper-v2':
            self.mutation_scale = 0.1
        else:
            self.mutation_scale = 0.4

        # Genetic memory size
        self.individual_bs = 8000

        # Variation operator statistics
        self.opstat = cla.opstat
        self.opstat_freq = cla.opstat_freq
        self.save_csv_freq = cla.save_csv_freq
        self.ignore_warmup_save = cla.ignore_warmup_save

        # Save Results
        self.state_dim = None  # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.action_high = None  # To be initialised externally
        self.action_low = None  # To be initialised externally
        self.save_foldername = cla.logdir
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)