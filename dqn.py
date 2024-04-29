import numpy as np
from encoder import SbertEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import utils
from torch.distributions import Categorical
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import time
from torch.distributions import MultivariateNormal
import copy
import crafter_env
from pathlib import Path



class WordGRU(nn.Module):
    def __init__(self, msg_hdim, msg_edim, device, vocab_size=1000):
        super().__init__()
        self.msg_hdim = msg_hdim
        self.msg_edim = msg_edim
        self.device = device

        self.emb = nn.Embedding(vocab_size, self.msg_edim, padding_idx=0)
        self.rnn = nn.GRU(self.msg_edim, self.msg_hdim, batch_first=True)

    def forward(self, messages, messages_len):  
        B, S = messages.shape
        
        embeds = torch.zeros(B, self.msg_hdim).to(self.device)
        zero_len = (messages_len == 0).squeeze()
        messages_emb = self.emb(messages)
        
        if B == 1:
            if zero_len:
                return embeds
            packed_input = pack_padded_sequence(messages_emb[~zero_len][0], messages_len[~zero_len, 0].cpu()[0], enforce_sorted=False, batch_first=True) 
        else:
            packed_input = pack_padded_sequence(messages_emb[~zero_len], messages_len[~zero_len, 0].cpu(), enforce_sorted=False, batch_first=True) 
        
        _, hidden = self.rnn(packed_input)
        
        embeds[~zero_len] = hidden[0]
        
        return embeds

class LanguageEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, device, type='gru', input_type='goal'):
        super().__init__()
        self.type = type
        self.vocab = vocab
        self.device = device
        
        assert input_type in ['goal', 'state'] # goal or state based encoder
        self.input_type = input_type
        
        if type == 'gru':  
            self.word_embedding = nn.Embedding(vocab, hidden_size)
            self.lang_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        elif type == 'pretrained_sbert':
            self.sbert_encoder = SbertEncoder(hidden_size, device)
            self.sbert_encoder.to(device)            
        elif type == 'wordgru':
            self.lang_encoder = WordGRU(hidden_size, msg_edim=64, vocab_size=vocab, device=device)
    

    def forward(self, obs):
        if self.type == 'gru':
            embedded_obs = self.word_embedding(obs)
            embedding, _ = self.lang_encoder(embedded_obs)  # shape is B x seq x hidden
            last_index = (obs != 0).sum(1).long() - 1 # How many non-padding indices are there?
            last_index = torch.maximum(last_index, torch.zeros_like(last_index))
            # TODO: THIS COMPUTATION IS WRONG IF YOU USE LANG OBS + GOAL
            B = len(last_index)
            embedding = embedding[range(B), last_index]  # Choose the embedding corresponding to the last non-padding input
        elif self.type == 'pretrained_sbert':
            embedding = self.sbert_encoder(obs)
        elif self.type == 'wordgru':
            message_len = (obs != 0).long().sum(-1, keepdim=True)
            embedding = self.lang_encoder(obs, message_len)
                              
        return embedding


class Encoder(nn.Module):

    def __init__(self, obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim=None):
        super().__init__()
        # assert use_image or use_language, "must provide either img or lang obs"
        self.use_image = use_image
        self.use_language = use_language_state or use_goal
        self.use_language_state = use_language_state
        self.use_goal = use_goal
        # if other_dim == 0:
        #     other_dim += 1
        self.other_dim = other_dim 
        self.device = device
        self.hidden_dim = hidden_dim
        self.goal_encoder_type = goal_encoder_type
        obs_inputs = 0
        if self.use_image:
            obs_inputs += 1
            self.img_encoder = ImageEncoder(obs_shape, hidden_dim)
        if self.use_language:
            if self.use_goal: # Note: sbert 'text_obs' encoding includes goal already
                obs_inputs += 1
                if goal_encoder_type == 'sbert':
                    self.lang_goal_encoder = LanguageEncoder(vocab, hidden_dim, device, type='pretrained_sbert', input_type='goal')
                else:
                    self.lang_goal_encoder = LanguageEncoder(vocab, hidden_dim, device, type=goal_encoder_type, input_type='goal') 
            if self.use_language_state == 'sbert':
                if self.use_goal and goal_encoder_type == 'sbert':
                    pass
                else:
                    self.lang_state_encoder = LanguageEncoder(vocab, hidden_dim, device, type='pretrained_sbert', input_type='state')
                    obs_inputs += 1
            elif self.use_language_state == 'conv':
                obs_inputs += 1
                self.lang_state_encoder = ImageEncoder((1,7,9), hidden_dim, semantic=True, vocab=vocab)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * obs_inputs + int(other_dim), hidden_dim), nn.ReLU())

    def forward(self, obs):
        obs_encoded = []
        if self.use_image:
            try:
                obs_encoded.append(self.img_encoder((obs['obs'])))
            except:
                import pdb; pdb.set_trace()
        if self.use_language:
            if self.use_goal and self.goal_encoder_type == 'sbert' and self.use_language_state == 'sbert':
                state_and_goal = torch.cat([obs['text_obs'], obs['goal']], dim=-1)  
                obs_encoded.append(self.lang_goal_encoder(state_and_goal))  
            else:
                if self.use_goal:
                    obs_encoded.append(self.lang_goal_encoder(obs['goal']))
                if self.use_language_state:
                    obs_encoded.append(self.lang_state_encoder(obs['text_obs'])) 
        if self.other_dim > 0:
            if len(obs['other'].shape) == 1:
                obs_encoded.append(obs['other'].unsqueeze(-1))
            else:
                obs_encoded.append(obs['other'])
        
        obs_encoded = torch.cat(obs_encoded, -1)
        obs_encoded = self.mlp(obs_encoded)
        return obs_encoded


class ImageEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim, semantic=False, vocab=None):
        super().__init__()
        self.embedding = None
        self.semantic = semantic # language embeddings instead of pixels
        if semantic:
            self.vocab = vocab
            self.feature_dim = 64 * 4 * 6
            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 2, stride=1), nn.ReLU(),
                nn.Conv2d(32, 64, 2, stride=1), nn.ReLU(),
                nn.Conv2d(64, 64, 2, stride=1), nn.ReLU())
        else:
            if obs_shape[-1] == 84:
                self.feature_dim = 64 * 7 * 7
            elif obs_shape[-1] == 128:
                self.feature_dim = 128 * 72
            else:
                raise NotImplementedError

            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU())
        self.linear = nn.Linear(self.feature_dim, out_dim)
        

    def forward(self, obs):
        if self.semantic:
            obs = obs[..., :63].view(obs.shape[0], 1, 7, 9) # obs is 7 x 9 tiles 
            # Normalize by vocab size
            obs = obs / self.vocab
        elif obs.dtype == torch.uint8:
            # Pixels
            obs = obs.type(torch.float32)
            obs = obs / 255.
        if self.embedding is not None:
            B, C, H, W = obs.shape
            obs = self.embedding(obs.int().flatten(1)).reshape((B, C, H, W, -1))
            obs = obs.moveaxis(-1, 2)
            obs = obs.reshape((B, -1, H, W))
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        h = F.normalize(h, p=2, dim=1)
        h = self.linear(h)
        return h


class DuelCritic(nn.Module):
    def __init__(self, encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim):
        super().__init__()

        self.encoder = encoder

        self.V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, 1))
        
        self.A = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, num_actions))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.encoder(obs)
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q
    
    def forward_encoded(self, h):
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q


class DQNAgent:
    def __init__(self, obs_shape, num_actions,  device, lr,
                 critic_target_tau, critic_target_update_every_steps, train_eps_min, train_eps_max,
                 train_eps_decay_steps, reward_scale, eval_eps, update_every_steps,
                 use_tb, use_wandb,  hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, other_dim, finetune_settings, 
                 other_model_prob, debug, cfg, **kwargs):
        self.num_actions = num_actions
        self.critic_target_update_every_steps = critic_target_update_every_steps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.train_eps_min = train_eps_min
        self.train_eps_max = train_eps_max
        self.train_eps_decay_steps = train_eps_decay_steps
        self._reward_scale = reward_scale
        self.eval_eps = eval_eps
        self.device = device
        self.log = use_tb or use_wandb
        self.other_model_prob = other_model_prob
        self.metrics = {}
        self.encoder = Encoder(obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim)
        self.critic = DuelCritic(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target = DuelCritic(self.encoder, obs_shape, num_actions,
                                    hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.decay_started = False
        self.decay_started_step = None
        self.cfg = cfg

        if not finetune_settings == 'train_all':
            if 'linear' in finetune_settings:
                layers = [4]
            elif 'critic' in finetune_settings:
                layers = [0, 2, 4] 
            finetune_params = []
            for layer in layers:
                finetune_params += [self.critic.get_parameter(f'V.{layer}.weight'), self.critic.get_parameter(f'V.{layer}.bias'), self.critic.get_parameter(f'A.{layer}.weight'), self.critic.get_parameter(f'A.{layer}.bias')]
            self.opt = torch.optim.Adam(finetune_params, lr=lr)
        else:
            self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target.train()
        self.train()

    def new_opt(self, lr):
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def new_tau(self, tau):
        self.critic_target_tau = tau

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def compute_train_eps(self, step):
        if self.decay_started:
            step_unit = (self.train_eps_max - self.train_eps_min) / self.train_eps_decay_steps
            step_since_decay = step - self.decay_started_step
            train_eps = max(0, self.train_eps_max - step_unit * step_since_decay)
            return max(self.train_eps_min, train_eps)
        else:
            return self.train_eps_max

    def preprocess_obs(self, obs):
        """
        Input obs is a dictionary, with an image at 'obs' and language at 'goal'
        """
        preprocessed_obs = {}
        for k, v in obs.items():
            preprocessed_obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        return preprocessed_obs

    
    def act(self, obs, step, eval_mode, other_model=None):
        # For calling act on 'other model' 
        if other_model == -1:
            obs = self.preprocess_obs(obs)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()
            return action
            
        train_eps = self.compute_train_eps(step)
        eps = self.eval_eps if eval_mode else train_eps
        if np.random.rand() < eps:
            if other_model is not None and np.random.rand() < self.other_model_prob:
                action = other_model.act(obs, step, eval_mode=eval_mode, other_model=-1)
                self.rand_action = 'other'
            else:
                action = np.random.randint(self.num_actions)
                self.rand_action = 'random'
        else:
            self.rand_action = 'policy'
            obs = self.preprocess_obs(obs)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()

        return action

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        # print(f'obs = {obs.keys()}, action = {action.size()}, reward = {reward.size()}, discount = {discount.size()}')
        # print(obs['inv_status'].size(), '*************', obs['inv_status'])
        if not hasattr(self, '_reward_scale'):
            self._reward_scale = 1
        reward *= self._reward_scale
        
        with torch.no_grad():
            # share critic, critic_target encoder
            encoder_output = self.encoder(next_obs)
                
            next_action = self.critic.forward_encoded(encoder_output).argmax(dim=1).unsqueeze(1)
            next_Qs = self.critic_target.forward_encoded(encoder_output)
            next_Q = next_Qs.gather(1, next_action).squeeze(1)
            target_Q = reward + discount * next_Q

        # get current Q estimates
        Qs = self.critic(obs)
        Q = Qs.gather(1, action.unsqueeze(1)).squeeze(1)
        critic_loss = F.smooth_l1_loss(Q, target_Q)
            
        if self.log:
            # print('this is DQN', Q.mean().item(), reward.mean().item(), critic_loss.item())
            metrics['q'] = Q.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['train_eps'] = self.compute_train_eps(step)
            try:
                new_metrics_dict = self.log_time()
                metrics.update(new_metrics_dict)
            except Exception as e:
                pass
        
        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        # Log the gradient norms
        if self.log:
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    metrics['grad_norm_' + name] = param.grad.norm().item()
        
        self.opt.step()
        # print('one step done (DQN)')

        return metrics

    def log_time(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.log()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.log()

    def update(self, replay_iter, step, train_env, use_extr_rew=False, cfg = None):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
            
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        self.metrics = self.update_critic(batch, step)

        if step % self.critic_target_update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return self.metrics

    def load_and_save_cache(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.load_and_save_cache()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.load_and_save_cache()



class Actor(nn.Module):
    def __init__(self, encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim):
        super().__init__()

        self.encoder = encoder

        # self.V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                        nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
        #                        nn.ReLU(), nn.Linear(hidden_dim, 1))
        
        self.A = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, num_actions), 
                            #    nn.Softmax(dim=-1)
                            )

        self.apply(utils.weight_init) # TODO

    def forward(self, obs):
        h = self.encoder(obs)
        # v = self.V(h)
        a = self.A(h)
        # q = v + a - a.mean(1, keepdim=True)
        # return q
        return a
    
    def forward_encoded(self, h):
        # v = self.V(h)
        a = self.A(h)
        # q = v + a - a.mean(1, keepdim=True)
        # return q
        return a



class Critic(nn.Module):
    def __init__(self, encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim):
        super().__init__()

        self.encoder = encoder

        self.Value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, 1))
        
        # self.A = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                        nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
        #                        nn.ReLU(), nn.Linear(hidden_dim, num_actions))

        # self.apply(utils.weight_init) # TODO

    def forward(self, obs):
        h = self.encoder(obs)
        v = self.Value(h)
        # a = self.A(h)
        # q = v + a - a.mean(1, keepdim=True)
        # return q
        return v

    
    def forward_encoded(self, h):
        v = self.Value(h)
        # a = self.A(h)
        # q = v + a - a.mean(1, keepdim=True)
        # return q
        return v

class ActorCritic:
    def __init__(self, obs_shape, num_actions,  device, lr,
                 critic_target_tau, critic_target_update_every_steps, train_eps_min, train_eps_max,
                 train_eps_decay_steps, reward_scale, eval_eps, update_every_steps,
                 use_tb, use_wandb,  hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, other_dim, finetune_settings, 
                 other_model_prob, debug, cfg, **kwargs):
        self.num_actions = num_actions
        self.critic_target_update_every_steps = critic_target_update_every_steps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.train_eps_min = train_eps_min
        self.train_eps_max = train_eps_max
        self.train_eps_decay_steps = train_eps_decay_steps
        self._reward_scale = reward_scale
        self.eval_eps = eval_eps
        self.device = device
        self.log = use_tb or use_wandb
        self.other_model_prob = other_model_prob
        self.metrics = {}
        self.encoder = Encoder(obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim)
        self.critic = Critic(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target = Critic(self.encoder, obs_shape, num_actions,
                                    hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.actor = Actor(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.decay_started = False
        self.decay_started_step = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.timesteps_per_batch = 32               # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 16 
        self.discount = 0.95

        if not finetune_settings == 'train_all':
            if 'linear' in finetune_settings:
                layers = [4]
            elif 'critic' in finetune_settings:
                layers = [0, 2, 4] 
            finetune_params = []
            for layer in layers:
                finetune_params += [self.critic.get_parameter(f'V.{layer}.weight'), self.critic.get_parameter(f'V.{layer}.bias'), self.critic.get_parameter(f'A.{layer}.weight'), self.critic.get_parameter(f'A.{layer}.bias')]
            self.opt = torch.optim.Adam(finetune_params, lr=lr)
        else:
            self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target.train()
        self.train()

    def new_opt(self, lr):
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def new_tau(self, tau):
        self.critic_target_tau = tau

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def compute_train_eps(self, step):
        if self.decay_started:
            step_unit = (self.train_eps_max - self.train_eps_min) / self.train_eps_decay_steps
            step_since_decay = step - self.decay_started_step
            train_eps = max(0, self.train_eps_max - step_unit * step_since_decay)
            return max(self.train_eps_min, train_eps)
        else:
            return self.train_eps_max

    def preprocess_obs(self, obs):
        """
        Input obs is a dictionary, with an image at 'obs' and language at 'goal'
        """
        preprocessed_obs = {}
        for k, v in obs.items():
            preprocessed_obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        return preprocessed_obs

    

    
    def act(self, obs, step, eval_mode, other_model=None):

        self.rand_action = 'policy'
        # action_probs = self.actor(obs)
        # action = np.random.choice(np.arange(self.num_actions), p=action_probs.detach().numpy().ravel())
        obs = self.preprocess_obs(obs)
        # print(f'obs = {obs}')
        action_probs = self.actor(obs)
        action_probs_softmax = torch.softmax(action_probs, dim=-1)


        dist = Categorical(action_probs_softmax)
        # Sample an action from the distribution
        action = dist.sample().item()
        # print(action, type(action))
        # print('****************')
        return action



    def get_action(self, obs):
        self.rand_action = 'policy'
        obs = self.preprocess_obs(obs)
        action_probs = self.actor(obs)
        action_probs_softmax = torch.softmax(action_probs, dim=-1)


        dist = Categorical(action_probs_softmax)
        # Sample an action from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = {}
        batch_next_obs = {}
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. Note that obs is short for observation. 
            # time_step, _ = self.env.reset()
            # print('it is while loop of rollout (actor critic)')

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                #     self.env.render()

                t += 1 # Increment timesteps ran this batch so far
                # with torch.no_grad():
                # # Track observations in this batch
                preprocessed = self.preprocess_obs(time_step.observation)
                for key, value in preprocessed.items():
                    # print('current key is :', key, 't is :', t)
                    if t == 1:
                        if key != 'inv_status':
                            batch_obs[key] = value
                    else:
                        if key != 'inv_status':
                            batch_obs[key] = torch.cat((batch_obs[key], value), dim = 0)
                    
                # print('batch_obs', batch_obs.keys(), batch_obs['obs'].size())
    
    
                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(time_step.observation)
                time_step = self.env.step(action)

                preprocessed = self.preprocess_obs(time_step.observation)
                for key, value in preprocessed.items():
                    # print('current key is :', key, 't is :', t)
                    if t == 1:
                        if key != 'inv_status':
                            batch_next_obs[key] = value
                    else:
                        if key != 'inv_status':
                            batch_next_obs[key] = torch.cat((batch_next_obs[key], value), dim = 0)
                    
                # print('We are in rollout')
                # print(type(time_step.observation['obs']), time_step.observation['obs'].shape)
                # Track recent reward, action, and action log probability
                ep_rews.append(time_step.reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_rews.append(time_step.reward)

                # If the environment tells us the episode is terminated, break
                if time_step.observation['goal_success'] == 1:
                    break

            # Track episodic lengths and rewards
            # batch_rews.append(ep_rews)
            # print('batch rews lenghs is :', len(batch_rews), batch_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_acts = torch.tensor(batch_acts).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs).to(self.device)
        batch_rews = torch.tensor(batch_rews).to(self.device)
        # batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4


        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_next_obs
    

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        # obs, action, log_probs, reward, next_obs = self.rollout()
        if not hasattr(self, '_reward_scale'):
            self._reward_scale = 1
        reward *= self._reward_scale

        # No target network in standard Actor-Critic, direct next_obs evaluation
        with torch.no_grad():
            encoder_output = self.encoder(next_obs)

            # For the Critic
            value_next = self.critic.forward_encoded(encoder_output).squeeze(1)
            # print(f'value next size = {value_next.size()}, reward size = {reward.size()}')
            target_value = reward + self.discount * value_next

        # Critic Update
        value = self.critic(obs).squeeze(1)
        critic_loss = F.smooth_l1_loss(value, target_value)
        # Actor Update
        # We use the critic's value as an approximation of the state value for computing advantages
        advantage = (reward + self.discount * value_next.detach()) - value.detach()
        logits = self.actor(obs)
        log_probs = -F.cross_entropy(logits, action.squeeze(), reduction = 'none')
        # log_probs = torch.log(self.actor(obs) + 0.0001)
        # action_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        actor_loss = -(log_probs * advantage.detach()).mean()
        # Logging metrics
        if self.log:
            # print('we are logging')
            # print('logging items are:', actor_loss.item(), critic_loss.item(), value.mean().item(), advantage.mean().item())
            metrics['actor_loss'] = actor_loss.item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['value'] = value.mean().item()
            metrics['advantage'] = advantage.mean().item()

        # Zero gradients before backward pass
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()

        # Backward pass for both Actor and Critic
        actor_loss.backward()
        critic_loss.backward()

        # Perform a step with both optimizers
        self.actor_opt.step()
        self.critic_opt.step()
        print('one step done (ActorCritic)')

        return metrics



 

    def log_time(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.log()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.log()

    def update(self, replay_iter, step, train_env, use_extr_rew=False, cfg = None):
        metrics = dict()
        self.env = train_env
        if step % self.update_every_steps != 0:
            return metrics
            
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        self.metrics = self.update_critic(batch, step)
        return self.metrics

    def load_and_save_cache(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.load_and_save_cache()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.load_and_save_cache()


# class PPO:
#     def __init__(self, obs_shape, num_actions,  device, lr,
#                  critic_target_tau, critic_target_update_every_steps, train_eps_min, train_eps_max,
#                  train_eps_decay_steps, reward_scale, eval_eps, update_every_steps,
#                  use_tb, use_wandb,  hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, other_dim, finetune_settings, 
#                  other_model_prob, debug, **kwargs):
#         """
#             Initializes the PPO model, including hyperparameters.

#             Parameters:
#                 env - the environment to train on.
#                 hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

#             Returns:
#                 None
#         """
#         # Make sure the environment is compatible with our code
#         # assert(type(env.observation_space) == gym.spaces.Box)
#         # assert(type(env.action_space) == gym.spaces.Box)

#         # Initialize hyperparameters for training with PPO
#         self.num_actions = num_actions
#         self.critic_target_update_every_steps = critic_target_update_every_steps
#         self.critic_target_tau = critic_target_tau
#         self.update_every_steps = update_every_steps
#         self.train_eps_min = train_eps_min
#         self.train_eps_max = train_eps_max
#         self.train_eps_decay_steps = train_eps_decay_steps
#         self._reward_scale = reward_scale
#         self.eval_eps = eval_eps
#         self.device = device
#         self.log = use_tb or use_wandb
#         self.other_model_prob = other_model_prob
#         self.metrics = {}
#         self.encoder = Encoder(obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim)
#         self.critic = Critic(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
#         self.critic_target = Critic(self.encoder, obs_shape, num_actions,
#                                     hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
#         self.actor = Actor(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        
#         self._init_hyperparameters()

#         # Initialize optimizers for actor and critic
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

#         # Initialize the covariance matrix used to query the actor for actions
#         self.cov_var = torch.full(size=(self.num_actions,), fill_value=0.5)
#         self.cov_mat = torch.diag(self.cov_var)

#         # This logger will help us with printing out summaries of each iteration
#         self.logger = {
#             'delta_t': time.time_ns(),
#             't_so_far': 0,          # timesteps so far
#             'i_so_far': 0,          # iterations so far
#             'batch_lens': [],       # episodic lengths in batch
#             'batch_rews': [],       # episodic returns in batch
#             'actor_losses': [],     # losses of actor network in current iteration
#         }
#         self.train()

#     def learn(self, total_timesteps):
#         torch.autograd.set_detect_anomaly(True)
#         """
#             Train the actor and critic networks. Here is where the main PPO algorithm resides.

#             Parameters:
#                 total_timesteps - the total number of timesteps to train for

#             Return:
#                 None
#         """
#         print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
#         print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
#         t_so_far = 0 # Timesteps simulated so far
#         i_so_far = 0 # Iterations ran so far
#         while t_so_far < total_timesteps:                                                                       # ALG STEP 2
#             # Autobots, roll out (just kidding, we're collecting our batch simulations here)
#             batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3
#             print('this is the beginning of learn')


# 			# Calculate how many timesteps we collected this batch
#             t_so_far += np.sum(batch_lens)

# 			# Increment the number of iterations
#             i_so_far += 1

#             # Logging timesteps so far and iterations so far
#             self.logger['t_so_far'] = t_so_far
#             self.logger['i_so_far'] = i_so_far

#             # Calculate advantage at k-th iteration
#             V, _ = self.evaluate(batch_obs, batch_acts)
#             A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

#             # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
#             # isn't theoretically necessary, but in practice it decreases the variance of 
#             # our advantages and makes convergence much more stable and faster. I added this because
#             # solving some environments was too unstable without it.
#             A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            

#             # This is the loop where we update our network for some n epochs
#             for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
#                 # Calculate V_phi and pi_theta(a_t | s_t)
#                 V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

#                 # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
#                 # NOTE: we just subtract the logs, which is the same as
#                 # dividing the values and then canceling the log with e^log.
#                 # For why we use log probabilities instead of actual probabilities,
#                 # here's a great explanation: 
#                 # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
#                 # TL;DR makes gradient ascent easier behind the scenes.
#                 ratios = torch.exp(curr_log_probs  - batch_log_probs.detach())

#                 # Calculate surrogate losses.
#                 surr1 = ratios * A_k
#                 surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

#                 # Calculate actor and critic losses.
#                 # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
#                 # the performance function, but Adam minimizes the loss. So minimizing the negative
#                 # performance function maximizes it.
#                 actor_loss = (-torch.min(surr1, surr2)).mean()
#                 critic_loss = nn.MSELoss()(V, batch_rtgs.detach())

     
#                 print('the error is here 2')
#                 # Calculate gradients and perform backward propagation for critic network
#                 self.critic_optim.zero_grad()
#                 critic_loss.backward()
#                 self.critic_optim.step()

#                 # Calculate gradients and perform backward propagation for actor network
#                 print('the error is here.')
#                 self.actor_optim.zero_grad()
#                 actor_loss.backward(retain_graph=True)
#                 self.actor_optim.step()
#                 # Log actor loss
#                 self.logger['actor_losses'].append(actor_loss.detach())

#             # Print a summary of our training so far
#             self._log_summary()

#             # Save our model if it's time
#             if i_so_far % self.save_freq == 0:
#                 torch.save(self.actor.state_dict(), './ppo_actor.pth')
#                 torch.save(self.critic.state_dict(), './ppo_critic.pth')
#             print('HOOOOORAY WE RAN ONE EPOCH FOR PPO')


#     def rollout(self):
#         """
#             Too many transformers references, I'm sorry. This is where we collect the batch of data
#             from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
#             of data each time we iterate the actor/critic networks.

#             Parameters:
#                 None

#             Return:
#                 batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
#                 batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
#                 batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
#                 batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
#                 batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
#         """
#         # Batch data. For more details, check function header.
#         batch_obs = {}
#         batch_acts = []
#         batch_log_probs = []
#         batch_rews = []
#         batch_rtgs = []
#         batch_lens = []

#         # Episodic data. Keeps track of rewards per episode, will get cleared
#         # upon each new episode
#         ep_rews = []

#         t = 0 # Keeps track of how many timesteps we've run so far this batch
#         # Keep simulating until we've run more than or equal to specified timesteps per batch
#         while t < self.timesteps_per_batch:
#             ep_rews = [] # rewards collected per episode

#             # Reset the environment. sNote that obs is short for observation. 
#             time_step, _ = self.env.reset()
#             done = False
#             print('it is while loop of rollout')

#             # Run an episode for a maximum of max_timesteps_per_episode timesteps
#             for ep_t in range(self.max_timesteps_per_episode):
#                 # If render is specified, render the environment
#                 # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
#                 #     self.env.render()

#                 t += 1 # Increment timesteps ran this batch so far
#                 with torch.no_grad():
#                 # Track observations in this batch
#                     preprocessed = self.preprocess_obs(time_step.observation)
#                 for key, value in preprocessed.items():
#                     # print('current key is :', key, 't is :', t)
#                     if t == 1:
#                         if key != 'inv_status':
#                             batch_obs[key] = value
#                     else:
#                         if key != 'inv_status':
#                             batch_obs[key] = torch.cat((batch_obs[key], value), dim = 0)
                    
#                 # print('batch_obs', batch_obs.keys(), batch_obs['obs'].size())
    
    
#                 # Calculate action and make a step in the env. 
#                 # Note that rew is short for reward.
#                 action, log_prob = self.get_action(time_step.observation)
#                 time_step = self.env.step(action)
#                 # print('We are in rollout')
#                 # print(type(time_step.observation['obs']), time_step.observation['obs'].shape)
#                 # Track recent reward, action, and action log probability
#                 ep_rews.append(time_step.reward)
#                 batch_acts.append(action)
#                 batch_log_probs.append(log_prob)

#                 # If the environment tells us the episode is terminated, break
#                 if time_step.observation['goal_success'] == 1:
#                     break

#             # Track episodic lengths and rewards
#             batch_lens.append(ep_t + 1)
#             batch_rews.append(ep_rews)

#         # Reshape data as tensors in the shape specified in function description, before returning
#         batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
#         batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
#         batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

#         # Log the episodic returns and episodic lengths in this batch.
#         self.logger['batch_rews'] = batch_rews
#         self.logger['batch_lens'] = batch_lens

#         return batch_obs, batch_acts, batch_log_probs, batch_rtgs.to(self.device), batch_lens

#     def compute_rtgs(self, batch_rews):
#         """
#             Compute the Reward-To-Go of each timestep in a batch given the rewards.

#             Parameters:
#                 batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

#             Return:
#                 batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
#         """
#         # The rewards-to-go (rtg) per episode per batch to return.
#         # The shape will be (num timesteps per episode)
#         batch_rtgs = []

#         # Iterate through each episode
#         for ep_rews in reversed(batch_rews):

#             discounted_reward = 0 # The discounted reward so far

#             # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
#             # discounted return (think about why it would be harder starting from the beginning)
#             for rew in reversed(ep_rews):
#                 discounted_reward = rew + discounted_reward * self.gamma
#                 batch_rtgs.insert(0, discounted_reward)

#         # Convert the rewards-to-go into a tensor
#         batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

#         return batch_rtgs



    
#     def act(self, obs, step, eval_mode, other_model = None):

#         self.rand_action = 'policy'
#         obs = self.preprocess_obs(obs)
#         action_probs = self.actor(obs)
#         action_probs_softmax = torch.softmax(action_probs, dim=-1)


#         dist = Categorical(action_probs_softmax)
#         # Sample an action from the distribution
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         # mean = self.actor(obs)
#         # mean = mean.to(self.device)
#         # self.cov_mat = self.cov_mat.to(self.device)
#         # # return action
#         # dist = MultivariateNormal(mean, self.cov_mat)

#         # # Sample an action from the distribution
#         # action = dist.sample()

#         # # Calculate the log probability for that action
#         # log_prob = dist.log_prob(action)

#         # Return the sampled action and the log probability of that action in our distribution
#         # print(action.detach(), type(action.detach()), type(action.detach().item()))
#         # print('***************')
#         return action.detach().item()

#     def get_action(self, obs):
#         self.rand_action = 'policy'
#         obs = self.preprocess_obs(obs)
#         action_probs = self.actor(obs)
#         action_probs_softmax = torch.softmax(action_probs, dim=-1)


#         dist = Categorical(action_probs_softmax)
#         # Sample an action from the distribution
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         # mean = self.actor(obs)
#         # mean = mean.to(self.device)
#         # self.cov_mat = self.cov_mat.to(self.device)
#         # # return action
#         # dist = MultivariateNormal(mean, self.cov_mat)

#         # # Sample an action from the distribution
#         # action = dist.sample()

#         # # Calculate the log probability for that action
#         # log_prob = dist.log_prob(action)

#         # Return the sampled action and the log probability of that action in our distribution
#         # print(action.detach(), type(action.detach()), type(action.detach().item()))
#         # print('***************')
#         # print(type(action), type(log_prob), action, log_prob)
#         return action.detach(), log_prob.detach()

#     def evaluate(self, obs, acts):
#         """
#             Estimate the values of each observation, and the log probs of
#             each action in the most recent batch with the most recent
#             iteration of the actor network. Should be called from learn.

#             Parameters:
#                 batch_obs - the observations from the most recently collected batch as a tensor.
#                             Shape: (number of timesteps in batch, dimension of observation)
#                 batch_acts - the actions from the most recently collected batch as a tensor.
#                             Shape: (number of timesteps in batch, dimension of action)

#             Return:
#                 V - the predicted values of batch_obs
#                 log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
#         """
#         # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
#         value = self.critic(obs).to(self.device)

#         # OLD
#         # # Calculate the log probabilities of batch actions using most recent actor network.
#         # # This segment of code is similar to that in get_action()
#         # mean = self.actor(obs)
#         # dist = MultivariateNormal(mean, self.cov_mat)
#         # # print('batch acts size =', batch_acts)
#         # log_probs = dist.log_prob(acts)
#         # NEW
#         action_probs = self.actor(obs)
#         action_probs_softmax = torch.softmax(action_probs, dim=-1)


#         dist = Categorical(action_probs_softmax)
#         # Sample an action from the distribution
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         # Return the value vector V of each observation in the batch
#         # and log probabilities log_probs of each action in the batch
#         return value, log_prob
    
#     """
#      self.rand_action = 'policy'
#         obs = self.preprocess_obs(obs)
#         action_probs = self.actor(obs)
#         action_probs_softmax = torch.softmax(action_probs, dim=-1)


#         dist = Categorical(action_probs_softmax)
#         # Sample an action from the distribution
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         # mean = self.actor(obs)
#         # mean = mean.to(self.device)
#         # self.cov_mat = self.cov_mat.to(self.device)
#         # # return action
#         # dist = MultivariateNormal(mean, self.cov_mat)

#         # # Sample an action from the distribution
#         # action = dist.sample()

#         # # Calculate the log probability for that action
#         # log_prob = dist.log_prob(action)

#         # Return the sampled action and the log probability of that action in our distribution
#         # print(action.detach(), type(action.detach()), type(action.detach().item()))
#         # print('***************')
#         # print(type(action), type(log_prob), action, log_prob)
#         """

#     def _init_hyperparameters(self):
#         """
#             Initialize default and custom values for hyperparameters

#             Parameters:
#                 hyperparameters - the extra arguments included when creating the PPO model, should only include
#                                     hyperparameters defined below with custom values.

#             Return:
#                 None
#         """
#         hyperparameters = {
# 				'timesteps_per_batch': 2048, 
# 				'max_timesteps_per_episode': 200, 
# 				'gamma': 0.99, 
# 				'n_updates_per_iteration': 10,
# 				'lr': 3e-4, 
# 				'clip': 0.2,
# 				'render': True,
# 				'render_every_i': 10
# 			  }

#         # Initialize default values for hyperparameters
#         # Algorithm hyperparameters
#         self.timesteps_per_batch = 256                 # Number of timesteps to run per batch
#         self.max_timesteps_per_episode = 128           # Max number of timesteps per episode
#         self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
#         self.lr = 0.005                                 # Learning rate of actor optimizer
#         self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
#         self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

#         # Miscellaneous parameters
#         self.render = True                              # If we should render during rollout
#         self.render_every_i = 10                        # Only render every n iterations
#         self.save_freq = 10                             # How often we save in number of iterations
#         self.seed = None                                # Sets the seed of our program, used for reproducibility of results

#         # Change any default values to custom values for specified hyperparameters
#         for param, val in hyperparameters.items():
#             exec('self.' + param + ' = ' + str(val))

#         # Sets the seed if specified
#         if self.seed != None:
#             # Check if our seed is valid first
#             assert(type(self.seed) == int)

#             # Set the seed 
#             torch.manual_seed(self.seed)
#             print(f"Successfully set seed to {self.seed}")

#     def preprocess_obs(self, obs):
#         """
#         Input obs is a dictionary, with an image at 'obs' and language at 'goal'
#         """
#         preprocessed_obs = {}
#         for k, v in obs.items():
#             preprocessed_obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
#         return preprocessed_obs


#     def _log_summary(self):
#         """
#             Print to stdout what we've logged so far in the most recent batch.

#             Parameters:
#                 None

#             Return:
#                 None
#         """
#         # Calculate logging values. I use a few python shortcuts to calculate each value
#         # without explaining since it's not too important to PPO; feel free to look it over,
#         # and if you have any questions you can email me (look at bottom of README)
#         delta_t = self.logger['delta_t']
#         self.logger['delta_t'] = time.time_ns()
#         delta_t = (self.logger['delta_t'] - delta_t) / 1e9
#         delta_t = str(round(delta_t, 2))

#         t_so_far = self.logger['t_so_far']
#         i_so_far = self.logger['i_so_far']
#         avg_ep_lens = np.mean(self.logger['batch_lens'])
#         avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
#         avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

#         # Round decimal places for more aesthetic logging messages
#         avg_ep_lens = str(round(avg_ep_lens, 2))
#         avg_ep_rews = str(round(avg_ep_rews, 2))
#         avg_actor_loss = str(round(avg_actor_loss, 5))

#         # Print logging statements
#         print(flush=True)
#         print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
#         print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
#         print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
#         print(f"Average Loss: {avg_actor_loss}", flush=True)
#         print(f"Timesteps So Far: {t_so_far}", flush=True)
#         print(f"Iteration took: {delta_t} secs", flush=True)
#         print(f"------------------------------------------------------", flush=True)
#         print(flush=True)

#         # Reset batch-specific logging data
#         self.logger['batch_lens'] = []
#         self.logger['batch_rews'] = []
#         self.logger['actor_losses'] = []




#     def update(self, replay_iter, step, train_env, use_extr_rew=False):
#         metrics = dict()
#         self.env = train_env

#         if step % self.update_every_steps != 0:
#             return metrics
            
#         batch = next(replay_iter)
#         batch = utils.to_torch(batch, self.device)

#         # self.metrics = self.update_critic(batch, step)
#         self.metrics = self.learn(total_timesteps = 2)
#         return self.metrics

#     def new_opt(self, lr):
#         self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
#     # def new_tau(self, tau):
#     #     self.critic_target_tau = tau

#     def train(self, training=True):
#         self.training = training
#         self.critic.train(training)

#     def compute_train_eps(self, step):
#         if self.decay_started:
#             step_unit = (self.train_eps_max - self.train_eps_min) / self.train_eps_decay_steps
#             step_since_decay = step - self.decay_started_step
#             train_eps = max(0, self.train_eps_max - step_unit * step_since_decay)
#             return max(self.train_eps_min, train_eps)
#         else:
#             return self.train_eps_max
        



class PPO:
    def __init__(self, obs_shape, num_actions,  device, lr,
                 critic_target_tau, critic_target_update_every_steps, train_eps_min, train_eps_max,
                 train_eps_decay_steps, reward_scale, eval_eps, update_every_steps,
                 use_tb, use_wandb,  hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, other_dim, finetune_settings, 
                 other_model_prob, debug, cfg = None, **kwargs):
        self.num_actions = num_actions
        self.critic_target_update_every_steps = critic_target_update_every_steps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.train_eps_min = train_eps_min
        self.train_eps_max = train_eps_max
        self.train_eps_decay_steps = train_eps_decay_steps
        self._reward_scale = reward_scale
        self.eval_eps = eval_eps
        self.device = device
        self.log = use_tb or use_wandb
        self.other_model_prob = other_model_prob
        self.metrics = {}
        self.encoder = Encoder(obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim)
        self.critic = Critic(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target = Critic(self.encoder, obs_shape, num_actions,
                                    hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.actor = Actor(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.decay_started = False
        self.decay_started_step = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.timesteps_per_batch = 64           # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 32
        self.discount = 0.95
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.gamma = 0.95
        self.cfg = cfg
        self.work_dir = Path.cwd()

        self.env = crafter_env.make(
                logdir=self.work_dir / 'train_episodes' / 'PPO_env',
                env_spec=self.cfg,
                save_video=False,
                frame_stack=4,
                seed=1,
                env_reward=True,
                use_wandb=True,
                debug=False,
                device='cuda')
        self.time_step, _ = self.env.reset()

        if not finetune_settings == 'train_all':
            if 'linear' in finetune_settings:
                layers = [4]
            elif 'critic' in finetune_settings:
                layers = [0, 2, 4] 
            finetune_params = []
            for layer in layers:
                finetune_params += [self.critic.get_parameter(f'V.{layer}.weight'), self.critic.get_parameter(f'V.{layer}.bias'), self.critic.get_parameter(f'A.{layer}.weight'), self.critic.get_parameter(f'A.{layer}.bias')]
            self.opt = torch.optim.Adam(finetune_params, lr=lr)
        else:
            self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target.train()
        self.train()

    def new_opt(self, lr):
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def new_tau(self, tau):
        self.critic_target_tau = tau

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def compute_train_eps(self, step):
        if self.decay_started:
            step_unit = (self.train_eps_max - self.train_eps_min) / self.train_eps_decay_steps
            step_since_decay = step - self.decay_started_step
            train_eps = max(0, self.train_eps_max - step_unit * step_since_decay)
            return max(self.train_eps_min, train_eps)
        else:
            return self.train_eps_max

    def preprocess_obs(self, obs):
        """
        Input obs is a dictionary, with an image at 'obs' and language at 'goal'
        """
        preprocessed_obs = {}
        for k, v in obs.items():
            preprocessed_obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        return preprocessed_obs

    

    
    def act(self, obs, step, eval_mode, other_model=None):

        self.rand_action = 'policy'
        obs = self.preprocess_obs(obs)
        action_probs = self.actor(obs)
        action_probs_softmax = torch.softmax(action_probs, dim=-1)
        dist = Categorical(action_probs_softmax)
        action = dist.sample().item()
        return action

    def get_action(self, obs):
        self.rand_action = 'policy'
        obs = self.preprocess_obs(obs)
        action_probs = self.actor(obs)
        action_probs_softmax = torch.softmax(action_probs, dim=-1)


        dist = Categorical(action_probs_softmax)
        # Sample an action from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def rollout(self):
        # Batch data. For more details, check function header.
        batch_obs = {}
        batch_next_obs = {}
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. Note that obs is short for observation. 
            # print('it is while loop of rollout')

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                #     self.env.render()
                if self.time_step.last():
                    self.time_step, _ = self.env.reset()
                t += 1 # Increment timesteps ran this batch so far
                # with torch.no_grad():
                # # Track observations in this batch
                preprocessed = self.preprocess_obs(self.time_step.observation)
                for key, value in preprocessed.items():
                    # print('current key is :', key, 't is :', t)
                    if t == 1:
                        if key != 'inv_status':
                            batch_obs[key] = value
                    else:
                        if key != 'inv_status':
                            batch_obs[key] = torch.cat((batch_obs[key], value), dim = 0)
                    
                # print('batch_obs', batch_obs.keys(), batch_obs['obs'].size())
    
    
                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(self.time_step.observation)
                self.time_step = self.env.step(action)

                # preprocessed = self.preprocess_obs(time_step.observation)
                # for key, value in preprocessed.items():
                #     # print('current key is :', key, 't is :', t)
                #     if t == 1:
                #         if key != 'inv_status':
                #             batch_next_obs[key] = value
                #     else:
                #         if key != 'inv_status':
                #             batch_next_obs[key] = torch.cat((batch_next_obs[key], value), dim = 0)
                    
                # print('We are in rollout')
                # print(type(time_step.observation['obs']), time_step.observation['obs'].shape)
                # Track recent reward, action, and action log probability
                ep_rews.append(self.time_step.reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_rews.append(self.time_step.reward)

                # If the environment tells us the episode is terminated, break
                if self.time_step.observation['goal_success'] == 1:
                    break

            # Track episodic lengths and rewards
            # batch_rews.append(ep_rews)
            # print('batch rews lenghs is :', len(batch_rews), batch_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_acts = torch.tensor(batch_acts).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs).to(self.device)
        batch_rews = torch.tensor(batch_rews).to(self.device)
        # batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4


        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_next_obs
    

    def update_critic(self, batch, step):
        metrics = dict()
        # obs, action, reward, discount, next_obs = batch
        obs, action, log_probs, reward, next_obs = self.rollout()
        if not hasattr(self, '_reward_scale'):
            self._reward_scale = 1
        reward *= self._reward_scale


        """
        This right here is PPO
        """
        advantage = np.zeros(len(reward), dtype=np.float32)

        values = self.critic(obs)
        for t in range(len(reward)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward)-1):
                a_t += discount*(reward[k] + self.gamma*values[k+1] - values[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.device)
        action_probs = self.actor(obs)
        action_probs_softmax = torch.softmax(action_probs, dim = -1)
        dist = Categorical(action_probs_softmax)

        critic_value = self.critic(obs).to(self.device)
        critic_value = torch.squeeze(critic_value)
        new_probs = dist.log_prob(action)
        prob_ratio = new_probs.exp() / log_probs.exp()
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
        returns  = advantage + values

        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()
        total_loss = actor_loss + 0.5 * critic_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()

        """
        Until here.
        """

        # # No target network in standard Actor-Critic, direct next_obs evaluation
        # with torch.no_grad():
        #     encoder_output = self.encoder(next_obs)

        #     # For the Critic
        #     value_next = self.critic.forward_encoded(encoder_output).squeeze(1)
        #     # print(f'value next size = {value_next.size()}, reward size = {reward.size()}')
        #     target_value = reward + self.discount * value_next

        # # Critic Update
        # value = self.critic(obs).squeeze(1)
        # critic_loss = F.smooth_l1_loss(value, target_value)
        # # Actor Update
        # # We use the critic's value as an approximation of the state value for computing advantages
        # advantage = (reward + self.discount * value_next.detach()) - value.detach()
        # logits = self.actor(obs)
        # log_probs = -F.cross_entropy(logits, action.squeeze(), reduction = 'none')
        # # log_probs = torch.log(self.actor(obs) + 0.0001)
        # # action_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        # actor_loss = -(log_probs * advantage.detach()).mean()
        # # Logging metrics
        if self.log:
            metrics['actor_loss'] = actor_loss.item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['value'] = values.mean().item()
            metrics['advantage'] = advantage.mean().item()

        # # Zero gradients before backward pass
        # self.actor_opt.zero_grad()
        # self.critic_opt.zero_grad()

        # # Backward pass for both Actor and Critic
        # actor_loss.backward()
        # critic_loss.backward()

        # # Perform a step with both optimizers
        # self.actor_opt.step()
        # self.critic_opt.step()
        print('one step is done (PPO_NEW)')

        return metrics



 

    def log_time(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.log()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.log()

    def update(self, replay_iter, step, train_env, use_extr_rew=False, cfg = None):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics
            
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        self.metrics = self.update_critic(batch, step)
        return self.metrics

    def load_and_save_cache(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.load_and_save_cache()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.load_and_save_cache()
