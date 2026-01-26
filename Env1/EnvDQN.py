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
        self.dt = 1.0 # hour

        # battery parameters
        self.E_bat = 10.0 # kWh
        self.eta_c = 0.95
        self.eta_d = 0.95

        # HVAC power levels (kW)
        self.hvac_power = [0.0,4.0,8.0]

        # Battery power (kW)
        self.P_ch = 2.0
        self.P_dis= 2.0

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
        return np.array([self.T/30,outdoor_temperature(h)/30, self.SOC, pv_profile(h), price_profile(h),np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)], dtype=np.float32)

    def step(self,action):
            hvac_action = action // 3
            bat_action = action % 3

            P_hvac = self.hvac_power[hvac_action]
            #battery charging constraints
            if bat_action ==1 :
                if self.SOC == 1.0 :
                      P_ch=0
                      P_dis=0
                else :
                    P_ch=1
                    P_dis=0
            elif bat_action==2:
                    if self.SOC == 0.0 :
                      P_ch=0
                      P_dis=0
                    else :
                     P_ch=0
                     P_dis=1
            else:
                P_ch=0
                P_dis=0

            T_out = outdoor_temperature(self.hour)
            self.T += self.dt / self.C * ((T_out - self.T) / self.R + P_hvac)

            # Battery dynamics
            self.SOC += self.dt / self.E_bat * (
                self.eta_c * P_ch - P_dis / self.eta_d
            )
            SOC_violation = max(0.0,-self.SOC)**2 +max(0.0,self.SOC -1.0)**2
            self.SOC = np.clip(self.SOC, 0.0, 1.0)

        # Power balance
            P_pv = pv_profile(self.hour) * 3.0
            self.P_grid = max(0.0, P_hvac + P_ch - P_dis - P_pv)

            price = price_profile(self.hour)
            cost = self.P_grid * price

        # Comfort penalty
            comfort_penalty = 0.0
            if self.T < self.T_min:
                 comfort_penalty = 10*(self.T_min - self.T)**2
            elif self.T >self.T_max:
                 comfort_penalty = 10*(self.T_max - self.T)**2
            else : 
                 comfort_penalty = 0.2*(self.T - self.T_set)**2

            reward = -2.0*cost -comfort_penalty - 10.0 * SOC_violation

            self.hour += self.dt
            done = self.hour == 24.0
            if done :
                 reward -= 10.0* (max(0,self.SOC - 0.4)**2 + max(0,0.6 - self.SOC)**2)

            return self.get_state(), reward, done, {}


# defining the DQN elements
# Network class definition for the qnetwork and the target network 
class Network(nn.Module):
    def __init__(self,state_dim, action_dim):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(state_dim,64),
              nn.ReLU(),
              nn.Linear(64,64),
              nn.ReLU(),
              nn.Linear(64,action_dim)
          )
    def forward(self,x):
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self, capacity= 10000):
        self.buffer = deque( maxlen=capacity)

    def push(self, s,a,r,s2,d):
        self.buffer.append((s,a,r,s2,d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype= torch.float32,device=device),
            torch.tensor(a , dtype= torch.int64,device=device),
            torch.tensor(r, dtype= torch.float32,device=device).unsqueeze(1),
            torch.tensor(s2,dtype=torch.float32,device=device),
            torch.tensor(d,dtype=torch.float32,device=device).unsqueeze(1))
    def __len__(self):
        return len(self.buffer)

# the DQN class

class DQN():
    def __init__(self,state_dim,action_dim):
        #networks and buffer definition
        self.q_net = Network(state_dim,action_dim).to(device)
        self.target_net= Network(state_dim,action_dim).to(device)
       
        self.buffer = ReplayBuffer()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr= 1e-3)

        # hyperparameters 
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay= 0.995
        self.batch_size = 64
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self,state):

        if np.random.rand()<self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_t=torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad() :
            q= self.q_net(state_t)
            return int(torch.argmax(q))
        
    def greedy_action(self,state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad() :
            q= self.q_net(state_t)
            return int(torch.argmax(q))

    def train_step(self):
        if len(self.buffer)< self.batch_size:
            return
        
        s,a,r,s2,d = self.buffer.sample(self.batch_size)
        # retrieving the corresponding Q values from the Q network for the corresponding actions taken
        q= self.q_net(s).gather(1,a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]

        target = r+ self.gamma*q_next*(1-d)
        loss = nn.MSELoss()(q,target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # update the target network
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def push_buffer(self, s, a , r , s2 , d):
        self.buffer.push(s, a ,r,s2,d)
    def epsilon_update(self):
        self.epsilon= max(self.epsilon_min, self.epsilon_decay*self.epsilon)
    def save(self, path="dqn_model.pth"):
        torch.save({"q_net": self.q_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "q_optim": self.optimizer.state_dict()}, path)
    def load(self, path="dqn_model.pth"):
        checkpoint = torch.load(path,map_location=device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['q_optim'])

        self.q_net.to(device)
        self.target_net.to(device)

if __name__ == '__main__':
    env = EnergyEnv()
    state = env.reset()
    agent = DQN(7,9)
    episode_num = 40000
    episode_rewards= []
    total_prices = []
    target_update= 200
    step_count=0
    for episode in range(episode_num):
        
        total_reward = 0
        total_price=0.0
        state= env.reset()
        done= False
    
        while not done:

            with torch.no_grad():
                action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.push_buffer(state, action , reward, next_state,done)
            agent.train_step()

            step_count+=1
            state = next_state
            total_price += env.P_grid * price_profile(max(0,env.hour -1))*env.dt
            total_reward += reward
            if step_count%target_update ==0 :
                agent.update_target()
        # update epsilon after each episode
        agent.epsilon_update()
        # append rewards
        episode_rewards.append(total_reward)
        total_prices.append(total_price)
        if episode %5 ==0 :
             print(f" Episode {episode},Total Reward : {total_reward:.2f}, total Price : {total_price:.2f}")

    print(f" Episode {episode},Total Reward : {total_reward:.2f}")
    agent.save()

    window = 50
    moving_avg_rewards = np.convolve(
        episode_rewards, 
        np.ones(window)/window, 
        mode="valid"
    )
    moving_avg_prices = np.convolve(
        total_prices,
        np.ones(window)/window,
        mode="valid"
    )

    figure, ax = plt.subplots(2,1, figsize=(8,10))
    ax[0].plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax[0].plot(range(window-1, len(episode_rewards)), moving_avg_rewards, label="Moving avg")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Total reward")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(total_prices, alpha=0.3, label="Episode total price")
    ax[1].plot(range(window-1, len(total_prices)), moving_avg_prices, label="Moving avg")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Total price")
    ax[1].legend()
    ax[1].grid()
    plt.show()

    state = env.reset()
    T_log= list()
    SOC_log = list()
    grid_log= list()
    price=0.0

    for t in range(24):
        with torch.no_grad():
            action = agent.greedy_action(state)

        state, reward, done , _ = env.step(action)

        price += env.P_grid * price_profile(max(0,env.hour -1))
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

    print(f"Total price with DQN agent: {price:.2f}")
