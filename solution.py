# tweaking hyperparams (main) version  
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode
import copy 


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
       
        concat = torch.cat([x,a], dim=1)

        value = self.net(concat)

        #####################################################################
        return value


class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor.
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])

        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)

        raw = self.net(x) 
        
        action = self.action_scale * torch.tanh(raw) + self.action_bias
        
        #####################################################################
        return action


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.9 # MDP discount factor, originally 0.99
    exploration_noise: float = 0.09  # epsilon for epsilon-greedy exploration (Matteo's note: eps here is the sigma of the N(0, sigma) noise) originally 0.1
    tau: float = 0.003 #idk originally 0.005
    noise_clip: float = 0.4 #why not originally 0.5
    actor_lr: float = 1e-3 #originally 1e-4
    critic_lr: float = 1e-4 #originally 1e-3
    num_layers: int = 2 
    num_units: int = 512

    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()
        self.d = 2 

        #####################################################################
        # TODO: initialize actor, critic and attributes

        self.train_step = 0

        #critic and actor networks 
        self.critic1 = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units) #Q_theta1
        self.critic2 = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units) #Q_theta2 
        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.num_units) #pi_phi

        #target networks
        self.critic1_target = copy.deepcopy(self.critic1) 
        self.critic2_target = copy.deepcopy(self.critic2) 
        self.actor_target = copy.deepcopy(self.actor) 

        #optimizers (not sure that it's needed but in the paper's github they did this so I'm doing it as well for now) 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr) 
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.critic_lr) 
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

        #device 
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.actor_target.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)


        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        #####################################################################
        # TODO: code training logic

        target_action = self.actor_target(next_obs) + torch.randn_like(self.actor_target(next_obs)) * self.exploration_noise
        target_action = torch.clamp(target_action, self.action_low, self.action_high)
        # if self.train_step % 500 == 0 : 
        #     print("------TRAIN ITERATION : " + str(self.train_step // 500) + " ------------")
        #     print("\n target_action :")
        #     print(type(target_action)) 
        #     print(target_action.size())
        #     print("\n")
        target_Q1 = self.critic1_target(next_obs, target_action) # 256, 1
        target_Q2 = self.critic2_target(next_obs, target_action)
        reward = reward.unsqueeze(1) #256, 1
        done = done.unsqueeze(1) #256, 1
        target_Q = reward + (1-done) * self.gamma * torch.min(target_Q1, target_Q2) #.detach() #256, 256

        current_Q1 = self.critic1(obs, action) 
        current_Q2 = self.critic2(obs, action)

        critic1_loss = nn.functional.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = nn.functional.mse_loss(current_Q2, target_Q.detach())
        
        self.critic1_optimizer.zero_grad() 
        critic1_loss.backward() 
        self.critic1_optimizer.step() 

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        actor_loss = -self.critic1(obs, self.actor(obs)).mean() 

        # if self.train_step % 500 == 0 : 
        #     print("\n actor_loss :")
        #     print(type(actor_loss)) 
        #     print(actor_loss.size())
        #     print("\n")

        self.actor_optimizer.zero_grad()
        actor_loss.backward() 
        self.actor_optimizer.step() 

        self.train_step += 1
        if self.train_step % self.d == 0:
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()) : 
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()) :
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()) : 
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0) 

            if train:
                noise = torch.randn_like(action) * self.exploration_noise
                action += noise 
        
        action = torch.clamp(action, self.action_low, self.action_high)

        action = action.cpu().numpy() #maybe remove 
        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        print("\n Running seed: " + str(seed) + "\n") #Matteo
        print(" currently running: hyperparams tweaking main \n")
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        print("\n") 
        print("------- START WARMUP ----------")
        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        print("\n") 
        print("------- START TRAIN ----------")
        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        print("\n") 
        print("------- START TEST ----------")
        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
