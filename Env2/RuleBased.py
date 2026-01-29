import numpy as np
from Env2 import Env

class RuleBasedAgent:
    def __init__(self,env):
        self.env = env

        # soc safety margin
        self.SOC_MIN = 0.3
        self.SOC_MAX = 0.7
        self.prev_bat = 0.0
        self.prev_hvac = 0.0
        # comfort deadband
        self.T_DEADBAND = 0.5

    #price classification
    def price_level(self, price): 
        if price <= 0.12: 
            return "low"
        elif price<=0.25:
            return "mid"
        else:
            return "high"
    
    def  hvac_control(self):
        T= self.env.T
        T_out = Env.outdoor_temperature(self.env.hour)
        price = Env.price_profile(self.env.hour)
        level = self.price_level(price)

        # hard comfort enforcement
        if T<self.env.T_min:
            return 1.0
        if T>self.env.T_max:
            return -1.0
        
        error = self.env.T_set- T

        #allow small drift at high prices
        if level == "high":
            error = np.clip(error, -self.T_DEADBAND, self.T_DEADBAND)
        
        hvac = np.clip(0.2*error,self.prev_hvac-0.1,self.prev_hvac+0.1)
        self.prev_hvac = hvac
        # check this later when you have a cooling system
        return hvac
    
    # battery control
    def battery_control(self):
        soc = self.env.SOC
        price = Env.price_profile(self.env.hour)
        level = self.price_level(price)
        pv = Env.pv_profile(self.env.hour)

        # soc protection
        if soc <= self.SOC_MIN:
            return -0.5
        if soc >= self.SOC_MAX:
            return 0.5
    
        # price based  decisions
        if level=='low':
            return -1.0 # charge
        elif level == 'high':
            return 1.0 #discharge
        else:
            # mid price : absorb PV only
            if pv>0.2:
                return -0.3
            return 0.0
    
    def ev_control(self):
        h= self.env.hour

        # EV not connected 
        if not (h >= self.env.t_arr or h<= self.env.t_dep):
            return 0.0
        
        soc= self.env.EV_SOC
        price = Env.price_profile(h)
        level = self.price_level(price)

        #time remaining
        if h>= self.env.t_arr:
            t_rem = (24-h) + self.env.t_dep
        else:
            t_rem = self.env.t_dep - h
        
        # minimum required power
        E_rem = max(0.0 ,1.0-soc) * self.env.E_EV_req
        P_min = E_rem /( t_rem*self.env.eta_EV + 1e-6)
        P_min = np.clip(P_min/self.env.P_EV_max,0.0,1.0)

        # price aware charging 
        if level =="low":
            ev= 1.0
        elif level=="high":
            ev = 0.0
        else:
            ev = max(0.5,P_min)
        
        return ev
    
    # main policy 

    def act(self):
        hvac= self.hvac_control()
        bat= self.battery_control()
        ev = self.ev_control()

        return np.array([hvac,bat,ev],dtype=np.float32)