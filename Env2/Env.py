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

# profile for the non-shiftable load (non-controllable)

def non_shiftable_profile(hour):
    #fridge_load
    fridge_load = np.random.choice([0.0,0.15],p=[0.7,0.3])
    # fridge + other loads
    if hour<=6:
        return 0.3+ fridge_load
    elif hour<=18:
        return 0.6+fridge_load
    else:
        return 1.0+fridge_load



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
        self.P_bat_max = 5.0

        # hvac max(kW)
        self.P_hvac_max = 20.0
        self.P_hvac_prev=0.0

        # comfort
        self.T_min = 21.0
        self.T_max= 25.0
        self.T_set = 22.0

        # EV parameters
        self.E_EV_req = 8.0      # kWh
        self.P_EV_max = 3.5     # kW
        self.eta_EV = 0.95
        self.t_arr = 18.0
        self.t_dep = 8.0 # next day
        self.EV_SOC = 0.0

        self.reset()
    


    def reset(self):
            self.hour =0.0
            self.T =22.0
            self.SOC = np.random.uniform(0.4,0.6)
            self.non_shiftable = [non_shiftable_profile(t) for t in np.arange(0,24,self.dt)]
            return self.get_state()

    def get_state(self):
        h = self.hour
        return np.array([self.T/30,outdoor_temperature(h)/30, self.non_shiftable[int((h/self.dt) -1)] ,self.EV_SOC,self.SOC, pv_profile(h), price_profile(h)/0.4,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)], dtype=np.float32)
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
            r_batteryLife = -0.02*abs(P_bat)*self.dt # battery usage limit violation

            # thermal regulation 
            P_hvac= (action[0]+1)/2 *self.P_hvac_max
            T_out = outdoor_temperature(self.hour)

            self.T += self.dt / self.C * ((T_out - self.T) / self.R + P_hvac)
            r_hvac = -0.05*(P_hvac-self.P_hvac_prev)**2 #penality for agressive changes
            self.P_hvac_prev= P_hvac 

            # Comfort penalty

            if self.T < self.T_min:
                r_comfort= -100*(self.T - self.T_min)**2
            elif self.T > self.T_max :
                r_comfort = -100*(self.T - self.T_max)**2
            else : 
                r_comfort = -2 *(self.T -self.T_set)**2

            # EV charging
            P_EV=0.0
            r_EV=0.0
            if self.hour<=self.t_dep or self.hour>=self.t_arr:
                action_EV = (action[2]+1)/2  # rescale to [0,1]
                P_EV = action_EV * self.P_EV_max
                self.EV_SOC += self.dt * self.eta_EV * P_EV/self.E_EV_req
                EV_soc_violation = (max(0,self.EV_SOC - 1)**2 + max(0,-self.EV_SOC)**2)
                self.EV_SOC = np.clip(self.EV_SOC, 0.0, 1.0)
                r_EV += -100.0*EV_soc_violation
            if self.hour == self.t_dep:
                r_EV += -100.0*(self.EV_SOC - 1)**2  # EV departs, SOC reset
                self.EV_SOC =np.random.uniform(0.0,0.1)  # next day arrival SOC


        # Power balance
            P_pv = pv_profile(self.hour) * 3.0
            idx= int (self.hour / self.dt)
            self.P_grid = max(0.0, P_hvac - P_bat + self.non_shiftable[idx]+ P_EV - P_pv)

            price = price_profile(self.hour)
            cost = self.P_grid * price*self.dt


            reward = -cost + r_comfort +r_hvac+r_soc + r_batteryLife + r_EV
   

            self.hour += self.dt
            done = self.hour == 24
            if done :
                reward -= 100.0* (max(0,self.SOC - 0.4)**2 + max(0,0.6 - self.SOC)**2) # final soc penalty
            return self.get_state(), reward, done, {}


