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
    
    # fridge + other loads
    if hour<=6:
        return 0.3
    elif hour<=18:
        return 0.6
    else:
        return 1.0


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
        self.P_hvac_max = 10.0 # kw
        self.hvac_COP = 3.0
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
        self.EV_SOC = 0.0 # intial SOc of the ev for the first episode, after that it will be set inside the charging course

        self.reset()
    


    def reset(self):
            self.hour =0.0
            self.T =22.0
            self.SOC = np.random.uniform(0.4,0.6)
            return self.get_state()

    def get_state(self):
        h = self.hour

        load = non_shiftable_profile(h)
        return np.array([self.T/30.0,outdoor_temperature(h)/20.0, load ,self.EV_SOC,self.SOC, pv_profile(h), price_profile(h)/0.4,np.sin(2*np.pi*h/24.0),np.cos(2*np.pi*h/24.0)], dtype=np.float32)
    def step(self,action):

            # battery model
            P_bat = action[1] * self.P_bat_max  # Positive = charge, negative = discharge

            if P_bat > 0:  # Charging
                delta_E = self.eta_c * P_bat * self.dt
                self.SOC += delta_E / self.E_bat
            else:  # Discharging
                delta_E = abs(P_bat) * self.dt / self.eta_d
                self.SOC -= delta_E / self.E_bat

            self.SOC = np.clip(self.SOC, 0.0, 1.0)


            soc_low_violation = max(0, 0.05 - self.SOC)  # Penalty if < 5%
            soc_high_violation = max(0, self.SOC - 0.95)  # Penalty if > 95%
            r_soc = -5.0 * (soc_low_violation + soc_high_violation)
            r_batteryLife = -0.1*abs(P_bat)*self.dt # battery usage limit violation

            # thermal regulation 
            P_hvac= action[0]*self.P_hvac_max
            T_out = outdoor_temperature(self.hour)

            self.T += self.dt / self.C * ((T_out - self.T) / self.R + self.hvac_COP*P_hvac)
            r_hvac = -0.01*(P_hvac-self.P_hvac_prev)**2 #penality for agressive changes
            self.P_hvac_prev= P_hvac 

            # Comfort penalty
            # Soft boundaries with deadband
            if self.T < self.T_min - 1.0:
                # Critical violation (< 20°C)
                r_comfort = -3.0 * (self.T - (self.T_min - 1.0))**2
            elif self.T > self.T_max + 1.0:
                # Critical violation (> 26°C)
                r_comfort = -3.0 * (self.T - (self.T_max + 1.0))**2
            elif self.T < self.T_min:
                # Minor violation (20-21°C) - linear penalty
                r_comfort = -1.0 * ( self.T_min- self.T)
            elif self.T > self.T_max:
                # Minor violation (25-26°C) - linear penalty
                r_comfort = -1.0 * (self.T - self.T_max)
            else:
                # Inside comfort zone [21-25°C] - NO PENALTY
                r_comfort = 0.0
            # EV charging
            P_EV=0.0
            r_EV=0.0
            if self.hour<=self.t_dep or self.hour>=self.t_arr:
                action_EV = (action[2]+1)/2  # rescale to [0,1]
                P_EV = action_EV * self.P_EV_max
                self.EV_SOC += self.dt * self.eta_EV * P_EV/self.E_EV_req
                EV_soc_violation = (max(0,self.EV_SOC - 1)**2 + max(0,-self.EV_SOC)**2)
                self.EV_SOC = np.clip(self.EV_SOC, 0.0, 1.0)
                r_EV -= 5.0*EV_soc_violation
            if abs(self.hour - self.t_dep) < 1e-6:
                r_EV -= 5.0*(self.EV_SOC - 1.0)**2  # EV departs, SOC reset
                self.EV_SOC =np.random.uniform(0.0,0.15)  # next day arrival SOC


        # Power balance
            P_pv = pv_profile(self.hour) * 3.0
            load = non_shiftable_profile(self.hour)
            self.P_grid = max(0.0, abs(P_hvac) + P_bat + load+ P_EV - P_pv)

            price = price_profile(self.hour)
            cost = self.P_grid *price*self.dt


            reward = -cost + r_comfort +r_hvac+r_soc + r_batteryLife + r_EV
            reward= np.clip(reward,-10.0,1.0)

            self.hour += self.dt
            done = int(self.hour) == 24
            info= {'cost':cost , 'r_comfort':r_comfort,'r_hvac':r_hvac,'r_soc':r_soc,'r_EV':r_EV,'P_hvac': P_hvac,'T':self.T, 'r_batteryLife':r_batteryLife , 'SOC':self.SOC,'P_grid':self.P_grid,'EV_SOC':self.EV_SOC}
            return self.get_state(), reward, done, info


    def sample_random_action(self):
            return np.random.uniform(-1.0,1.0,3)