import numpy as np
from collections import defaultdict,deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)

def pv_profile(hour):
    return max(0.0, np.sin((hour-6)/12 * np.pi))

def price_profile(hour):
    if hour<=6:
        return 0.1 # off peak
    elif hour<= 16 :
        return 0.2 # MID
    else :
        return 0.4 # peak 

def outdoor_temperature(hour):
    return 10 + 10*np.sin((hour-6)/24 * 2*np.pi)

# Environment implementation

class EnergyEnv:
    def __init__(self):
        # thermal parameters
        self.R = 2.0    # °C/kW
        self.C = 5.0    # kWh/ °C
        self.dt = 0.25 # hour

        # battery parameters
        self.E_bat = 10.0 # kWh
        self.eta_c = 0.95
        self.eta_d = 0.95


        # Battery power (kW)
        self.P_bat_max = 5.0
        # hvac max(kW)
        self.P_hvac_max = 20.0
        

        # comfort
        self.T_min = 21.0
        self.T_max= 25.0
        self.T_set = 22.0

        self.reset()
    


    def reset(self):
            self.hour =0
            self.T =22.0
            self.SOC = 0.5
            return self.get_state()

    def get_state(self):
        h = self.hour
        return np.array([self.T/30,outdoor_temperature(h)/30, self.SOC, pv_profile(h), price_profile(h)/0.4,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)], dtype=np.float32)

    def step(self,action):

            # battery model
            P_bat = action[1]*self.P_bat_max
            if P_bat <0 :
                          self.SOC += self.dt / self.E_bat * (
                self.eta_c * (-P_bat)
            )

            else:
                self.SOC -= self.dt / self.E_bat * (
                P_bat/self.eta_d
            )

            soc_violation = (max(0,self.SOC - 1)**2 + max(0,-self.SOC)**2)
            self.SOC = np.clip(self.SOC, 0.0, 1.0)

            r_soc = -100.0*soc_violation # charging limit violation

            # thermal regulation 
            P_hvac= (action[0]+1)/2 *self.P_hvac_max
            T_out = outdoor_temperature(self.hour)

            self.T += self.dt / self.C * ((T_out - self.T) / self.R + P_hvac)


            # Comfort penalty

            if self.T < self.T_min:
                r_comfort= -100*(self.T - self.T_min)**2
            elif self.T >self.T_max :
                r_comfort = -100*(self.T - self.T_max)**2
            else : 
                r_comfort = -2 *(self.T -self.T_set)**2

        # Power balance
            P_pv = pv_profile(self.hour) * 3.0
            self.P_grid = max(0.0, P_hvac -P_bat- P_pv)

            price = price_profile(self.hour)
            cost = self.P_grid * price * self.dt


            reward = -cost + r_comfort +r_soc
            self.hour += self.dt
            done = self.hour == 24
            if done : 
                reward -= 100.0* (max(0,self.SOC - 0.4)**2 + max(0,0.6 - self.SOC)**2) # final soc penalty

            return self.get_state(), reward, done, {}



    
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
    
    def sample(self,state):
        mu , std = self(state)
        normal = torch.distributions.Normal(mu,std)
        z= normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1-action.pow(2)+ 1e-6)
        log_prob = log_prob.sum(dim=1,keepdim=True)

        return action,log_prob
    

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
        self.target_entropy = -action_dim

        self.gamma = 0.99
        self.tau = 0.005
        self.buffer= ReplayBuffer()


    def save(self):
            torch.save({
            "actor": self.actor.state_dict(),
            "critic1":self.q1.state_dict(),
            "critic2": self.q2.state_dict(),
            "critic1_target": self.q1_target.state_dict(),
            "critic2_target": self.q2_target.state_dict(),
            "optimizerActor": self.actor_opt.state_dict(),
            "optimizerCritic1": self.q1_opt.state_dict(),
            "optimizerCritic2": self.q2_opt.state_dict(),
            },"SACparameters.pt")


    def load(self,path):
        checkpoint = torch.load(path,weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["critic1"])
        self.q2.load_state_dict(checkpoint["critic2"])
        self.q1_target.load_state_dict(checkpoint["critic1_target"])
        self.q2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_opt.load_state_dict(checkpoint['optimizerActor'])
        self.q2_opt.load_state_dict(checkpoint['optimizerCritic2'])
        self.q1_opt.load_state_dict(checkpoint['optimizerCritic1'])
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()

    def update(self,batch_size=256):
        if len(self.buffer)<batch_size : 
            return
        
        s , a , r,s2 , d = self.buffer.sample(batch_size)

        #critic update

        with torch.no_grad():
            a2 , logp2 = self.actor.sample(s2)
            q1_t = self.q1_target(s2,a2)
            q2_t = self.q2_target(s2,a2)
            q_target = torch.min(q2_t,q1_t) - self.alpha * logp2
            y = r + self.gamma*(1-d)*q_target
        
        q1= self.q1(s,a)
        q2= self.q2(s,a)

        q1_loss = nn.MSELoss()(q1,y)
        q2_loss= nn.MSELoss()(q2,y)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # Actor update

        a_new  ,logp = self.actor.sample(s)
        q1_new = self.q1(s,a_new)
        q2_new =self.q2(s,a_new)
        q_new= torch.min(q1_new,q2_new)

        actor_loss = (self.alpha *logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # target update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2,self.q2_target)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def soft_update(self,net , target_net):
        for p , tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1-self.tau)*tp.data)






if __name__ == '__main__':
    env = EnergyEnv()
    state = env.reset()
    agent = SACAgent(7,2)
    episode_num = 2000
    episode_rewards= []


    for episode in range(episode_num):

        total_reward = 0
        state= env.reset()
        done= False
        while not done:
            state_t = torch.tensor(state, dtype= torch.float32,device=device).unsqueeze(0)
            with torch.no_grad():
                action , _ = agent.actor.sample(state_t)
            action = action.squeeze(0).cpu().numpy()

            next_state, reward, done, _ =env.step(action)



            agent.buffer.push(state, action , reward, next_state,done)
            agent.update()


            state = next_state
            total_reward += reward

        # append rewards
        episode_rewards.append(total_reward) 
        print(f" Episode {episode},Total Reward : {total_reward:.2f}")


    agent.save()
    
    window = 50
    moving_avg = np.convolve(
        episode_rewards, 
        np.ones(window)/window, 
        mode="valid"
    )

    plt.figure()
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window-1, len(episode_rewards)), moving_avg, label="Moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid()
    plt.show()

    state = env.reset()

    T_log= list()
    SOC_log = list()
    grid_log= list()
    price=0.0
    done= False
    total_reward=0.0

    while not done:
        state_t = torch.tensor(state, dtype = torch.float32,device=device).unsqueeze(0)

        with torch.no_grad():
            action , _ = agent.actor.sample(state_t)
        action = action.squeeze(0).cpu().numpy()


        state, reward, done , _ = env.step(action)


        price += env.P_grid * price_profile(max(0,env.hour-1))*env.dt

        total_reward+=reward
        T_log.append(env.T)
        SOC_log.append(env.SOC)
        grid_log.append(env.P_grid)
    

    fig , ax = plt.subplots(3,1, figsize=(8,12))
    ax[0].plot(T_log)
    ax[0].axhline(21, linestyle="--")
    ax[0].axhline(25, linestyle="--")
    ax[0].set_ylabel("Indoor Temperature (°C)")
    ax[0].set_xlabel("Hour")
    ax[0].grid()
    
    
    ax[1].plot(SOC_log)
    ax[1].set_ylabel("Battery SOC")
    ax[1].set_xlabel("Hour")
    ax[1].grid()

    

    ax[2].plot(grid_log)
    ax[2].set_ylabel("Grid Power (kW)")
    ax[2].set_xlabel("Hour")
    ax[2].grid()

    plt.show()

    print(f"Total price with SAC agent: {price:.2f}")
    print(f"Total reward for the episode : { total_reward:.2f}")

   

