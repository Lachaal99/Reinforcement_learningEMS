from SACAgent import SACAgent
import Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
SEED  = 42 
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_STEPS= 1000
BATCH_SIZE = 256
NUM_EPISODES= 5000
EVAL_FREQ=100
SAVE_FREQ = 500



env = Env.EnergyEnv()

agent = SACAgent(state_dim=9,action_dim=3)

episode_rewards= []
episode_prices = []
eval_rewards = []
eval_prices=[]


def plot_episode_info(reward, cost , soc, soc_EV,P_hvac , T):
    fig ,ax = plt.subplots(3,2,figsize=(10,12))
    x= np.arange(len(reward)) * 0.25
    #reward plot
    ax[0,0].plot(x, reward,label= " rewards")
    ax[0,0].set_xlabel("hours")
    ax[0,0].set_ylabel("reward")
    ax[0,0].grid()
    ax[0,0].legend()

    # cost plot
    ax[0,1].plot(x,cost,label="cost")
    ax[0,1].set_xlabel("hours")
    ax[0,1].set_ylabel("price")
    ax[0,1].grid()
    ax[0,1].legend()

    #battery SOC plot

    ax[1,0].plot(x,soc,label= "battery state of charge")
    ax[1,0].set_xlabel("hours")
    ax[1,0].set_ylabel("soc battery")
    ax[1,0].grid()
    ax[1,0].legend()

    # EV soc plot

    ax[1,1].plot(x,soc_EV, label= " EV battery state of charge")
    ax[1,1].set_xlabel("hours")
    ax[1,1].set_ylabel("ev battery")
    ax[1,1].grid()
    ax[1,1].legend()

    #P_hvac

    ax[2,0].plot(x,P_hvac,label= " hvac power ")
    ax[2,0].set_xlabel("hours")
    ax[2,0].set_ylabel("hvac power")
    ax[2,0].grid()
    ax[2,0].legend()

    # temperature evolution

    ax[2,1].axhline(y=21.0, color='r', linestyle='--', alpha=0.5, label='T_min')
    ax[2,1].axhline(y=25.0, color='r', linestyle='--', alpha=0.5, label='T_max')
    ax[2,1].axhline(y=22.0, color='g', linestyle='--', alpha=0.5, label='T_set')
    ax[2,1].plot(x, T, label='Actual T', color='blue', linewidth=2)
    ax[2,1].set_xlabel('hours')
    ax[2,1].set_ylabel(" °C")
    ax[2,1].grid()
    ax[2,1].legend()

    ax[0,0].set_ylabel("Reward")
    ax[0,1].set_ylabel("Cost ($)")
    ax[1,0].set_ylabel("Battery SOC (%)")
    ax[1,1].set_ylabel("EV SOC (%)")
    ax[2,0].set_ylabel("HVAC Power (kW)")
    ax[2,1].set_ylabel("Temperature (°C)")

    ax[0,0].set_ylabel("Reward")
    ax[0,1].set_ylabel("Cost ($)")
    ax[1,0].set_ylabel("Battery SOC (%)")
    ax[1,1].set_ylabel("EV SOC (%)")
    ax[2,0].set_ylabel("HVAC Power (kW)")
    ax[2,1].set_ylabel("Temperature (°C)")

    plt.show()


def evaluate_policy(env,agent, num_episodes=5):
    """evaluate learned policy"""
    rewards=[]
    prices= []
    soc = []
    soc_ev = []
    P_hvac = []
    T= []
    single_rewards = []
    single_cost= []

    for ep in range(num_episodes):
        state= env.reset()
        ep_price = 0
        ep_reward =0
        done = False
        while not done : 
            action = agent.select_action(state,evaluate=True)
            next_state, reward, done ,info = env.step(action)
            ep_reward += reward
            ep_price += info.get('cost',0.0)
            state= next_state
            if ep == num_episodes-1:
                soc.append(info.get("SOC",0.0))
                soc_ev.append(info.get("EV_SOC",0.0))
                P_hvac.append(info.get("P_hvac",0.0))
                T.append(info.get("T",0.0))
                single_rewards.append(reward)
                single_cost.append(info.get("cost",0.0))
        rewards.append(ep_reward)
        prices.append(ep_price)
    plot_episode_info(single_rewards,single_cost,soc,soc_ev,P_hvac,T)

    return np.mean(rewards), np.mean(prices)

print("starting training ....")

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward =0.0
    total_price = 0.0
    if (episode +1)%20==0:
        print("=============================================================================================================")
        print("=============================================================================================================")
        print("Hour | cost | r_comfort | r_hvac | r_soc | r_batteryLife | r_EV")
    while not done : 

        if len(agent.buffer)<WARMUP_STEPS:
            action = env.sample_random_action()
        else:
            action = agent.select_action(state,evaluate=False)
        
        next_state, reward,done ,info = env.step(action)

        agent.buffer.push(state,action,reward,next_state,done)

        if len(agent.buffer)>= BATCH_SIZE:
            agent.update()

        total_reward +=reward
        total_price +=info.get('cost',0.0)
        state = next_state

        if (episode +1)%20 ==0 :
            print(f"{env.hour} | {info.get('cost',0.0):.3f} | {info.get('r_comfort',0.0):.3f} | {info.get('r_hvac',0.0):.3f} | {info.get('r_soc',0.0):.3f} | {info.get('r_batteryLife',0):.3f} | {info.get('r_EV',0.0):.3f} ")
    episode_rewards.append(total_reward)
    episode_prices.append(total_price)
    # Periodic evaluation
    if (episode + 1) % EVAL_FREQ == 0:
        eval_r, eval_p = evaluate_policy(env=env,agent=agent)
        print(f"\n=== Episode {episode+1}/{NUM_EPISODES} ===")
        print(f"Train - Reward: {total_reward:.2f}, Price: {total_price:.2f}")
        print(f"Eval  - Reward: {eval_r:.2f}, Price: {eval_p:.2f}")
        print(f"Alpha: {agent.alpha.item():.4f}, Buffer: {len(agent.buffer)}")
    # Periodic checkpoint
    if (episode + 1) % SAVE_FREQ == 0:
        agent.save(f'sac_ep{episode+1}.pt')
        print(f"Checkpoint saved at episode {episode+1}")
    


# Plotting
window = 50
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Rewards
axes[0].plot(episode_rewards, alpha=0.3, label='Episode reward', color='blue')
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                 label=f'{window}-episode MA', color='red', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title('Training Rewards')

# Prices
axes[1].plot(episode_prices, alpha=0.3, label='Episode price', color='green')
if len(episode_prices) >= window:
    moving_avg = np.convolve(episode_prices, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(episode_prices)), moving_avg,
                 label=f'{window}-episode MA', color='orange', linewidth=2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Total Price ($)')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_title('Energy Costs')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Training curves saved to 'training_curves.png'")
plt.show()
