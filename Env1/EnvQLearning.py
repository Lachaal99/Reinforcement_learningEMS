import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def discretize(value, bins):
    return np.digitize(value, bins) - 1

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
        self.hvac_power = [0.0, 2.0 ,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0]

        # Battery power (kW)
        self.P_ch = 2.0
        self.P_dis= 2.0

        # comfort
        self.T_min = 21.0
        self.T_max= 25.0
        self.T_set = 22.0

        self.reset()


    def reset(self):
            self.hour =0.0
            self.T_out = outdoor_temperature(self.hour)
            self.T =22.0
            self.SOC = 0.5
            return self.get_state()

    def get_state(self):
            T_bin = discretize(self.T, [20,21,22,23,24,25])
            T_out_bin = discretize(self.T_out, [20,21,22,23,24,25])
            SOC_bin = discretize(self.SOC,[0.2,0.4,0.6,0.8])
            PV_bin = discretize(pv_profile(self.hour), [0.2,0.6])
            price_bin = discretize(price_profile(self.hour), [0.15,0.3])
            return(T_bin,T_out_bin,SOC_bin,PV_bin,price_bin,self.hour)

    def step(self,action):
            hvac_action = action//3
            bat_action = action%3

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

            self.T_out = outdoor_temperature(self.hour)
            self.T += self.dt / self.C * ((self.T_out - self.T) / self.R + P_hvac)

            # Battery dynamics
            self.SOC += self.dt / self.E_bat * (
                self.eta_c * P_ch - P_dis / self.eta_d
            )

            SOC_violation = max(0.0, -self.SOC)**2 + max(0.0, self.SOC - 1.0)**2
            self.SOC = np.clip(self.SOC, 0.0, 1.0)

        # Power balance
            P_pv = pv_profile(self.hour) * 3.0
            self.P_grid = max(0.0, P_hvac + P_ch - P_dis - P_pv)

            price = price_profile(self.hour)
            cost = self.P_grid * price

        # Comfort penalty
            comfort_penalty = 0.0
            if self.T < self.T_min:
                 comfort_penalty = 100*(self.T_min - self.T)**2
            elif self.T >self.T_max:
                 comfort_penalty = 100*(self.T_max - self.T)**2
            else : 
                 comfort_penalty = 2*(self.T - self.T_set)**2
                

            reward = -cost -comfort_penalty - 100.0 * SOC_violation

            self.hour += self.dt
            done = self.hour == 24.0
            if done :
                 reward -= 100.0* (max(0,self.SOC - 0.4)**2 + max(0,0.6 - self.SOC)**2)

            return self.get_state(), reward, done, {}

class Q_learning: 
    def __init__(self):
          self.Q_table = defaultdict(float)
          self.alpha = 0.1
          self.gamma = 0.95
          self.epsilon= 1.0
          self.epsilon_min = 0.05
          self.epsilon_decay = 0.995

    def choose_action(self,state):
        if np.random.rand()<self.epsilon: 
             return np.random.randint(0,33)
        else: 
             q_vals = [self.Q_table[(state,a)] for a in range(33)]
             return int(np.argmax(q_vals))
    
    def greedy_action(self,state):
         q_vals = [self.Q_table[(state,a)] for a in range(33)]
         return int(np.argmax(q_vals))
    
    def update(self,action,state,next_state,reward):
        best_next_q = max(self.Q_table[(next_state,a)] for a in range(33))
        self.Q_table[(state, action)] += self.alpha * (
            reward + self.gamma * best_next_q - self.Q_table[(state, action)]
        )
    def epsilon_update(self):
         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



if __name__ == '__main__':
    env = EnergyEnv()
    state = env.reset()
    agent = Q_learning()
    episode_num = 20000
    episode_rewards= []
    for episode in range(episode_num):
        
        total_reward = 0
        state= env.reset()
        done= False
    
        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.update(action,state,next_state,reward)
            state = next_state
            total_reward += reward
 
        episode_rewards.append(total_reward)
        if episode%100==0 : 
             print(f" Episode {episode},Total Reward : {total_reward:.2f}")
        agent.epsilon_update()
    print(f" Episode {episode},Total Reward : {total_reward:.2f}")
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

    for t in range(24):
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

print(f"Total price with QL agent: {price:.2f}")

