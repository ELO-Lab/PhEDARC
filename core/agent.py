import numpy as np
from core import mod_neuro_evo as utils_ne
from core.mod_utils import *
from core import replay_memory
from core import darc as darc
from core import replay_memory
from parameters import Parameters
from torch.optim import Adam
from tqdm import tqdm
import random
from copy import deepcopy


class Agent:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Initialize population
        self.pop = []
        self.buffer = []
        for _ in range(args.pop_size):
            agent = darc.GeneticAgent(args)
            self.pop.append(agent)

        # Initialize stable elite mechanism
        if self.args.stable:
            self.stable_elite = darc.GeneticAgent(args)
            self.stable_elite_reward = 0
            self.stable_elite_idx = -1

        # Initialize RL operator
        self.darc_op = darc.DARC(args)
        
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size, args.device)

        self.evolver = utils_ne.SSNE(self.args, self.darc_op.critic1, self.darc_op.critic2, self.evaluate, self.replay_buffer)

        self.best_train_reward = 0.0
        self.warm_up = self.args.warm_up

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = 0

    def evaluate(self, agent: darc.GeneticAgent, is_render=False, is_action_noise=False,
                 store_transition=True, net_index=None):
        total_reward = 0.0

        state = self.env.reset()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if self.args.render and is_render: self.env.render()
            
            action = agent.actor.select_action(self.env, np.array(state), self.num_frames, self.warm_up)

            if is_action_noise:
                noise = np.random.normal(0, self.args.action_high * self.args.expl_noise, size=self.args.action_dim)
                action += noise
                action = np.clip(action, self.args.action_low, self.args.action_high)

            # Simulate one step in environment
            next_state, reward, done, info = self.env.step(action.flatten())
            total_reward += reward

            transition = (state, action, next_state, reward, float(done))
            if store_transition:
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)

            state = next_state
        if store_transition: 
            self.num_games += 1

        return {'reward': total_reward}

    def train_darc(self, agent1: darc.GeneticAgent, agent2: darc.GeneticAgent):
        # Initialize RL actors, target actors and their optimizer
        rl_agent1 = deepcopy(agent1)
        rl_agent2 = deepcopy(agent2)
        actor1_target = deepcopy(agent1.actor)
        actor2_target = deepcopy(agent2.actor)
        actor1_optim = Adam(rl_agent1.actor.parameters(), lr=1e-3)
        actor2_optim = Adam(rl_agent2.actor.parameters(), lr=1e-3)

        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in tqdm(range(int(self.gen_frames * self.args.frac_frames_train) // (self.args.pop_size//4))):
                batch = self.replay_buffer.sample(self.args.batch_size)

                self.darc_op.update_parameters(
                    batch, rl_agent1.actor, rl_agent2.actor, actor1_target, actor2_target, actor1_optim, actor2_optim)

        if self.args.verbose_rl_eval:
            self.evaluate_rl_trainer(agent1, rl_agent1)
            self.evaluate_rl_trainer(agent2, rl_agent2)

        return rl_agent1, rl_agent2

    def evaluate_rl_trainer(self, original_agent: darc.GeneticAgent, new_rl_agent: darc.GeneticAgent):
        test_score_g = 0
        trials = 5
        for eval in range(trials):
            episode = self.evaluate(original_agent, is_render=False, is_action_noise=False, store_transition=False)
            test_score_g += episode['reward']
        test_score_g /= trials

        test_score_r = 0
        for eval in range(trials):
            episode = self.evaluate(new_rl_agent, is_render=False, is_action_noise=False, store_transition=False)
            test_score_r += episode['reward']
        test_score_r /= trials

        if self.args.opstat:
            if self.args.verbose_crossover:
                print("==================== RL Trainer ======================")
                print("Before PG:", test_score_g)
                print("After PG:", test_score_r)

    def check_stable_elite(self, elite_indexs, all_fitnesses):
        if self.stable_elite_idx == -1:
            self.stable_elite_idx = elite_indexs[0]
            self.stable_elite = deepcopy(self.pop[elite_indexs[0]])
            stable_score = 0
            stable_score += all_fitnesses[elite_indexs[0]]
            for eval in range(4):
                episode = self.evaluate(self.stable_elite, is_render=True, is_action_noise=False, store_transition=True)
                stable_score += episode['reward']
            stable_score /= 5.0
            self.stable_elite_reward = stable_score
            print("New elite rewards:", stable_score)
            print("True elite has been replaced")

        elif self.stable_elite_idx != elite_indexs[0]:
            stable_score = 0
            stable_score += all_fitnesses[elite_indexs[0]]
            for eval in range(4):
                episode = self.evaluate(self.pop[elite_indexs[0]], is_render=True, is_action_noise=False, store_transition=True)
                stable_score += episode['reward']
            stable_score /= 5.0
            print("New elite rewards:", stable_score)
            if self.stable_elite_reward <= stable_score:
                self.stable_elite_idx = elite_indexs[0]
                self.stable_elite = deepcopy(self.pop[elite_indexs[0]])
                self.stable_elite_reward = stable_score
                print("True elite has been replaced")
            else:
                print("True elite has not been replaced")

        elif self.stable_elite_idx == elite_indexs[0]:
                print("True elite has not been replaced")
        
    def train(self):
        self.iterations += 1
        self.gen_frames = 0

        if self.num_frames < self.warm_up:
            print("Warm-up phase")
        
        # ========================== EVALUATION ===========================
        # Evaluate parents genomes/individuals
        rewards = np.zeros(self.args.pop_size)
        for i, net in enumerate(self.pop):
            for _ in range(self.args.num_evals):
                episode = self.evaluate(net, is_render=False, is_action_noise=False, net_index=i)
                rewards[i] += episode['reward']

        rewards /= self.args.num_evals
        all_fitness = rewards
        best_train_fitness = np.max(all_fitness)

        prRed("Population fitness: {}".format(all_fitness))

        test_score = 0
        test_reward = []

        # Init elite indexs
        num_elites = int(self.args.pop_size * self.args.elite_fraction)
        if num_elites < 1:
            num_elites = 1
        elite_indexs = list(range(num_elites))

        if self.num_frames >= self.warm_up:
            # ========================== EVOLUTION  ==========================
            # NeuroEvolution's selection and recombination step
            elite_indexs = self.evolver.epoch(self.pop, all_fitness)
        
            # ========================== STABLE ELITE AND TEST SCORE  ==========================
            if self.args.stable:
                # Check if new elite is stable elite and calculate its test score
                self.check_stable_elite(elite_indexs, all_fitness)

                for eval in range(self.args.num_test_evals):
                    episode = self.evaluate(self.stable_elite, is_render=True, is_action_noise=False, store_transition=False)
                    test_score += episode['reward']
                    test_reward.append(episode['reward'])
                test_score /= self.args.num_test_evals

            else:
                # Validation test for NeuroEvolution champion
                champion = self.pop[np.argmax(all_fitness)]

                for eval in range(self.args.num_test_evals):
                    episode = self.evaluate(champion, is_render=True, is_action_noise=False, store_transition=False)
                    test_score += episode['reward']
                    test_reward.append(episode['reward'])
                test_score /= self.args.num_test_evals

            print('Test rewards: ', test_reward)

            # ========================== DARC ===========================
            # Train DARC (almost) half a population
            non_elite_indexs = list(set(range(self.args.pop_size)) - set(elite_indexs))
            pg_train_indexs = random.sample(non_elite_indexs, len(self.pop)//2 - ((len(self.pop)//2) % 2))

            for i in range(0, len(pg_train_indexs), 2):
                self.pop[pg_train_indexs[i]], self.pop[pg_train_indexs[i+1]] = self.train_darc(
                                                    self.pop[pg_train_indexs[i]], self.pop[pg_train_indexs[i+1]])

        # -------------------------- Collect statistics --------------------------
        return {
            'best_train_fitness': best_train_fitness,
            'test_score': test_score,
            'elite_index': elite_indexs[0],
        }