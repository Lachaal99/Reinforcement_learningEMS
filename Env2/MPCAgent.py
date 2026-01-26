import numpy as np 
import cvxpy as cp 

class MPCAgent():
    def __init__(self, env, horizon=24):
        self.env = env
        self.horizon = horizon
        self.P_hvac = cp.Variable(horizon)
        self.P_bat = cp.Variable(horizon)
        self.P_EV = cp.Variable(horizon)

        self.T = cp.Variable(horizon+1)
        self.SOC = cp.Variable(horizon+1)
        self.SOC_EV = cp.Variable(horizon+1)
        self.constraints= []

    def optimize(self, state):
        T0, SOC0, SOC_EV0 = state[0], state[1], state[2]
        dt = self.env.dt
        R = self.env.R
        C = self.env.C
        eta_c = self.env.eta_c
        eta_d = self.env.eta_d
        E_bat = self.env.E_bat
        P_bat_max = self.env.P_bat_max
        P_hvac_max = self.env.P_hvac_max
        E_EV_req = self.env.E_EV_req
        P_EV_max = self.env.P_EV_max
        eta_EV = self.env.eta_EV
        t_arr = self.env.t_arr
        t_dep = self.env.t_dep

        price_profile = [self.env.price_profile(t) for t in np.arange(0,24,dt)]

        # reset constraints
        self.constraints += [self.T[0] == T0]
        self.constraints += [self.SOC[0] == SOC0]
        self.constraints += [self.SOC_EV[0] == SOC_EV0]

        for t in range(self.horizon):
            # Thermal dynamics
            non_shiftable_load = self.env.non_shiftable[int(t*dt)]
            self.constraints += [
                self.T[t+1] == self.T[t] + dt/(R*C)*( - (self.T[t] - 30.0) + R*(self.P_hvac[t] + non_shiftable_load))
            ]
            # Battery dynamics
            self.constraints += [
                self.SOC[t+1] == self.SOC[t] + dt/E_bat*( eta_c*cp.pos(self.P_bat[t]) - 1/eta_d*cp.neg(self.P_bat[t]) )
            ]
            # EV dynamics
            if (t*dt >= t_arr) or (t*dt < t_dep):
                self.constraints += [
                    self.SOC_EV[t+1] == self.SOC_EV[t] + dt/E_EV_req * eta_EV * cp.pos(self.P_EV[t])
                ]
            else:
                self.constraints += [
                    self.SOC_EV[t+1] == self.SOC_EV[t]
                ]
            # Constraints on power and comfort
            self.constraints += [ 0 <= self.P_hvac[t], self.P_hvac[t] <= P_hvac_max ]
            self.constraints += [ -P_bat_max