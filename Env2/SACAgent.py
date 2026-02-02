import numpy as np
from collections import defaultdict,deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)


class ReplayBuffer:
    def __init__(self, capacity= 10000):
        self.buffer = deque( maxlen=capacity)

    def push(self, s,a,r,s2,d):
        self.buffer.append((s,a,r,s2,d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype= torch.float32,device=device),
            torch.tensor(a , dtype= torch.float32,device=device),
            torch.tensor(r, dtype= torch.float32,device=device).unsqueeze(1),
            torch.tensor(s2,dtype=torch.float32,device=device),
            torch.tensor(d,dtype=torch.float32,device=device).unsqueeze(1)
        )
    

    def __len__(self):
        return len(self.buffer)

# define the actor class

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256,action_dim)
        self.log_std = nn.Linear(256,action_dim)

        self.LOG_STD_MIN =-20
        self.LOG_STD_MAX = 2
    
    def forward(self,state):
        x= self.net(state)
        mu = self.mu(x)
        log_std= self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN,self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu , std
    
    def sample(self, state):
        mu, std = self(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action= torch.tanh(z)  # In [-1, 1]
    
         # Compute log_prob for action_raw BEFORE rescaling
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
    
        return action, log_prob
    

# critic class

class Critic(nn.Module):
    def __init__(self,state_dim , action_dim):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(state_dim + action_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self,state,action):
        x= torch.cat([state,action],dim = 1 )
        return self.net(x)
    
# Defining the SAC agent

class SACAgent:
    def __init__(self , state_dim, action_dim): 
        self.actor = Actor(state_dim,action_dim).to(device)
        self.q1= Critic(state_dim, action_dim).to(device)
        self.q2= Critic(state_dim, action_dim).to(device)
        self.q1_target = Critic(state_dim,action_dim).to(device)
        self.q2_target= Critic(state_dim, action_dim).to(device)

        # loading the critic parameters into the target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # optimization hyperparameters
        self.actor_opt = optim.Adam(self.actor.parameters(),lr=3e-4)
        self.q1_opt=optim.Adam(self.q1.parameters(),lr=3e-4)
        self.q2_opt= optim.Adam(self.q2.parameters(),lr=3e-4)

        # entropy temperature
        self.log_alpha = torch.zeros(1,requires_grad=True,device= device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr= 3e-4)
        self.target_entropy = action_dim

        self.gamma = 0.99
        self.tau = 0.005
        self.buffer= ReplayBuffer()

    def save(self,path):
            torch.save({
            "actor": self.actor.state_dict(),
            "critic1":self.q1.state_dict(),
            "critic2": self.q2.state_dict(),
            "critic1_target": self.q1_target.state_dict(),
            "critic2_target": self.q2_target.state_dict(),
            "log_alpha":self.log_alpha.detach().cpu(),
            "optimizerActor": self.actor_opt.state_dict(),
            "optimizerCritic1": self.q1_opt.state_dict(),
            "optimizerCritic2": self.q2_opt.state_dict(),
            "optimizerAlpha":self.alpha_opt.state_dict(),
            },path)
    
    
    def load(self,path, load_optimizers=True,inference_only=False):
        checkpoint = torch.load(path,map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        if not inference_only:
            self.q1.load_state_dict(checkpoint["critic1"])
            self.q2.load_state_dict(checkpoint["critic2"])
            self.q1_target.load_state_dict(checkpoint["critic1_target"])
            self.q2_target.load_state_dict(checkpoint["critic2_target"])

            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(device))
            if load_optimizers:

                self.actor_opt.load_state_dict(checkpoint['optimizerActor'])
                self.q2_opt.load_state_dict(checkpoint['optimizerCritic2'])
                self.q1_opt.load_state_dict(checkpoint["optimizerCritic1"])
                self.alpha_opt.load_state_dict(checkpoint['optimizerAlpha'])

            self.actor.to(device)
            self.q1.to(device)
            self.q2.to(device)
            self.q1_target.to(device)
            self.q2_target.to(device)

    def update(self,batch_size=256):
        if len(self.buffer)<batch_size : 
            return
        
        s , a , r,s2 , d = self.buffer.sample(batch_size)

        #critic update

        with torch.no_grad():
            a2 , logp2 = self.actor.sample(s2)
            q1_t = self.q1_target(s2,a2)
            q2_t = self.q2_target(s2,a2)
            q_target = (torch.min(q2_t,q1_t) - self.alpha * logp2)
            y = r + self.gamma*(1-d)*q_target
        
        q1= self.q1(s,a)
        q2= self.q2(s,a)

        q1_loss = nn.SmoothL1Loss()(q1,y)
        q2_loss= nn.SmoothL1Loss()(q2,y)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q2_opt.step()

        # Actor update

        a_new  ,logp = self.actor.sample(s)
        q1_new = self.q1(s,a_new)
        q2_new =self.q2(s,a_new)
        q_new= torch.min(q1_new,q2_new)

        actor_loss = (self.alpha *logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        # GRADIENT CLIPPING 
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # target update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2,self.q2_target)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def soft_update(self,net, target_net):
        for p , tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1-self.tau)*tp.data)

    def select_action(self, state, evaluate=False):
        """Select action for environment interaction."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            if evaluate:
                mu, _ = self.actor(state_t)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state_t)
            return action.cpu().numpy()[0]