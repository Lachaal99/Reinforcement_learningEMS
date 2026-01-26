# MPC with discrete actions

import cvxpy as cp
import numpy as np
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
            self.T =22.0
            self.SOC = 0.5
            return self.get_state()

    def get_state(self):
            return(self.T,outdoor_temperature(self.hour),self.SOC,pv_profile(self.hour),price_profile(self.hour),self.hour)

    def step(self,action):



            P_hvac = action[0]
            P_ch=action[1]
            P_dis= action[2]

            T_out = outdoor_temperature(self.hour)
            self.T += self.dt / self.C * ((T_out - self.T) / self.R + P_hvac)

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

class MPCController:
    def __init__(self,env , horizon=24):
            self.env = env
            self.N = horizon

    def optimize(self,current_state):
        N= self.N
        dt = self.env.dt

        # decision variables
        P_hvac = cp.Variable(N)
        P_ch = cp.Variable(N)
        P_dis = cp.Variable(N)

        T= cp.Variable(N+1)
        SOC = cp.Variable(N+1)

        cost =0
        constraints = []
        
        T0 = current_state[0]
        SOC0 = current_state[2]
        
        constraints += [T[0] == T0]
        constraints += [SOC[0] == SOC0]

        hour0 = current_state[5]
        for k in range(N):
             h = int((hour0 + k * dt) % 24)

             T_out= outdoor_temperature(h)
             P_pv = pv_profile(h)*3.0
             price = price_profile(h)

             # dynamics
             constraints += [T[k+1] == T[k] + dt/self.env.C * 
                             ((T_out - T[k])/self.env.R + P_hvac[k])]
             constraints += [ SOC[k+1] == SOC[k] + dt/self.env.E_bat *
                             (self.env.eta_c * P_ch[k] - P_dis[k]/self.env.eta_d)]
             
             # constraints

             constraints += [ 
                  0<= P_hvac[k], P_hvac[k] <= 20.0,
                  0<= P_ch[k], P_ch[k] <= self.env.P_ch,
                  0<= P_dis[k],  P_dis[k] <= self.env.P_dis,
                  0<= SOC[k+1] , SOC[k+1]<= 1.0,
                  self.env.T_min <= T[k+1], T[k+1]<= self.env.T_max
             ]

             P_grid = cp.pos(P_hvac[k] + P_ch[k] - P_dis[k] - P_pv)

             cost += P_grid * price
             cost += 0.01 * (P_ch[k] + P_dis[k])
             cost += 2* cp.abs(T[k] - self.env.T_set)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver= cp.ECOS, verbose = False)

        return float(P_hvac.value[0]),float(P_ch.value[0]),float(P_dis.value[0])


if __name__ == '__main__':
     env = EnergyEnv()
     mpc = MPCController(env, horizon=24)

     state = env.reset()
     done = False
     total_reward = 0
     T_log= list()
     SOC_log = list()
     grid_log= list()
     total_cost=0.0
     
     while not done:
          P_hvac, P_ch , P_dis = mpc.optimize(state)
          print(f"hour {env.hour} : hvac power {P_hvac} , Battery charge {P_ch} , Battery discharge {P_dis}")

          T_log.append(env.T)
          SOC_log.append(env.SOC)
          action = [P_hvac, P_ch,P_dis]

          state,reward,done,_ = env.step(action)
          total_reward+= reward
          grid_log.append(env.P_grid)
          total_cost += env.P_grid * price_profile(env.hour)


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

     print(f"Total price with MPC agent: {total_cost:.2f}")
