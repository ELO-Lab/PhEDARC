import numpy as np, os, time, random
from core import mod_utils as utils, agent
import gym, torch
import argparse
import pickle
from parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Environment Choices: (Swimmer-v2) (HalfCheetah-v2) (Hopper-v2) ' +
                                 '(Walker2d-v2) (Ant-v2)', required=True, type=str)
parser.add_argument('--seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('--disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('--render', help='Render gym episodes', action='store_true')
parser.add_argument('--proximal_mut', help='Use safe mutation', action='store_true')
parser.add_argument('--phenotypic_mut', help='Use phenotypic differential mutation', action='store_true')
parser.add_argument('--distil', help='Use distilation crossover', action='store_true')
parser.add_argument('--distil_type', help='Use distilation crossover. Choices: (fitness) (distance)',
                    type=str, default='fitness')
parser.add_argument('--mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('--mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('--stable', help='Use stable mechanism', action='store_true')
parser.add_argument('--verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('--verbose_crossover', help='Make crossovers verbose', action='store_true')
parser.add_argument('--verbose_rl_eval', help='Make RL evaluation verbose', action='store_true')
parser.add_argument('--logdir', help='Folder where to save results', type=str, required=True)
parser.add_argument('--opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('--opstat_freq', help='Frequency (in generations) to store operator statistics', type=int, default=1)
parser.add_argument('--save_csv_freq', help='Frequency (in generations) to store score statistics', type=int, default=1)
parser.add_argument('--ignore_warmup_save', help='Start saving score after the warm-up period', action='store_true')
parser.add_argument('--save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('--next_save', help='Generation save frequency for save_periodic', type=int, default=200)
parser.add_argument('--next_frame_save', help='Actor save frequency', type=int, default=1000000)
parser.add_argument('--num_test_evals', help='Number of episodes to evaluate test score', type=int, default=5)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parameters = Parameters(parser)  # Inject the cla arguments in the parameters object
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')              # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')    # Initiate tracker

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.action_high = env.action_space.high[0]
    parameters.action_low = env.action_space.low[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(stdout=True)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    agent = agent.Agent(parameters, env)
    utils.prYellow('Running {} State_dim: {} Action_dim: {}'.format(
                                    parameters.env_name, parameters.state_dim, parameters.action_dim))

    next_frame_save = parameters.next_frame_save
    next_save = parameters.next_save

    gen = 1
    
    time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        utils.prGreen("Generation {}".format(gen))
        stats = agent.train()
        best_train_fitness = stats['best_train_fitness']
        erl_score = stats['test_score']
        elite_index = stats['elite_index']

        utils.prLightPurple('#Games: {} #Frames: {} Train_Max: {} Test_Score: {} Avg: {} ENV: {}'.format(
            agent.num_games, 
            agent.num_frames,
            round(best_train_fitness, 2) if best_train_fitness is not None else None,
            round(erl_score, 2) if erl_score is not None else None, 
            round(tracker.all_tracker[0][1], 2), 
            parameters.env_name))

        gen += 1

        print()
        # Log results periodically
        if agent.num_frames <= agent.warm_up:
            if not parameters.ignore_warmup_save:
                tracker.update([erl_score], agent.num_games)
                frame_tracker.update([erl_score], agent.num_frames)
                time_tracker.update([erl_score], time.time()-time_start)
        else:
                tracker.update([erl_score], agent.num_games)
                frame_tracker.update([erl_score], agent.num_frames)
                time_tracker.update([erl_score], time.time()-time_start)

        # Save Policy
        # Save stable elite policy if stable elite mechanism is selected
        if parameters.stable:
            if agent.num_frames > next_frame_save:
                if elite_index is not None:
                    print("Save stable")
                    torch.save(agent.stable_elite.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                    'evo_net_{}.pkl'.format(next_frame_save)))
                print('Stable elite is saved after the {}-th frames.'.format(next_frame_save))
                next_frame_save += parameters.next_frame_save

            if agent.num_games > next_save:
                next_save += parameters.next_save
                if elite_index is not None:
                    torch.save(agent.stable_elite.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                                'evo_net.pkl'))

                if parameters.save_periodic:
                    save_folder = os.path.join(parameters.save_foldername, 'models')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                    critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                    buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                    torch.save(agent.stable_elite.actor.state_dict(), actor_save_name)
                    torch.save(agent.rl_agent.critic.state_dict(), critic_save_name)
                    with open(buffer_save_name, 'wb+') as buffer_file:
                        pickle.dump(agent.rl_agent.buffer, buffer_file)

                print("Progress Saved")
        
        # If not, save the best elite of each generation 
        else:
            if agent.num_frames > next_frame_save:
                if elite_index is not None:
                    print("Save elite")
                    torch.save(agent.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                    'evo_net_{}.pkl'.format(next_frame_save)))
                print('Actor is saved after the {}-th frames.'.format(next_frame_save))
                next_frame_save += parameters.next_frame_save

            if agent.num_games > next_save:
                next_save += parameters.next_save
                if elite_index is not None:
                    torch.save(agent.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                                'evo_net.pkl'))

                if parameters.save_periodic:
                    save_folder = os.path.join(parameters.save_foldername, 'models')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                    critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                    buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                    torch.save(agent.pop[stats['elite_index']].actor.state_dict(), actor_save_name)
                    torch.save(agent.rl_agent.critic.state_dict(), critic_save_name)
                    with open(buffer_save_name, 'wb+') as buffer_file:
                        pickle.dump(agent.rl_agent.buffer, buffer_file)

                print("Progress Saved")