import gymnasium as gym
import random
import matplotlib.pyplot as plt
import numpy as np
from env import tf_env
from model import agent
from reinforcementagent import portfolio_valuation
from fe import valid, train
from env import positions, GymWrapper
from tf_agents.environments import tf_py_environment
from reward import calculate_reward

def create_env(start=0, finish=1000):
    env_valid = gym.make("TradingEnv",
                         name="BTCUSD",
                         df=valid[start:finish],
                         # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                         positions=positions,
                         # 0.01% per stock buy / sell (Binance fees)
                         trading_fees=0.01/100,
                         # 0.0003% per timestep (one timestep = 1h here)
                         borrow_interest_rate=0.0003/100,
                         verbose=1,
                         reward_function=lambda history: calculate_reward(
                             history, training=False)
                         )

    tf_env_valid = GymWrapper(gym_env=env_valid, df=valid, window_size=1000)
    tf_env_valid = tf_py_environment.TFPyEnvironment(tf_env_valid)
    return tf_env_valid


tf_env_valid = create_env()


def get_indexes(max_interval=200_000, total_time_steps=len(train)):
    max_start_index = total_time_steps - max_interval
    start = random.randint(0, max_start_index)
    finish = start + (max_interval - 1)
    return start, finish


plt.plot(portfolio_valuation)
plt.xlabel('Iteration')
plt.ylabel('USD')
plt.title('Portfolio valuation')
plt.show()

tf_env.envs[0]._gym_env.df
tf_env_valid.reset()

time_step = tf_env_valid.reset()
policy_state = agent.policy.get_initial_state(tf_env_valid.batch_size)

# Evaluation loop
done = False
iteration = 0
portfolio_valuations = []
actions_taken = []
window_iter = []
total_time_steps = len(valid)

for window in range(10):
    for iteration in range(1000):
        if done:
            done = False
            break
        action_step = agent.policy.action(time_step, policy_state)
        actions_taken.append(action_step)
        policy_state = action_step.state

        next_time_step = tf_env_valid.step(action_step.action)

        time_step = next_time_step
        policy_state = action_step.state

        done = time_step.is_last()

        eval_log = tf_env_valid.envs[0].logging_buffer()
        eval_portfolio_valuation = eval_log['portfolio_valuation']
        window_iter.append(eval_portfolio_valuation)
        print(
            f"Portfolio Valuation: {eval_portfolio_valuation} Iteration: {iteration}/{1000} Window: {window}/10", end="\r")

    portfolio_valuations.append(window_iter)
    start, finish = get_indexes(max_interval=1000, total_time_steps=len(valid))
    for env in tf_env_valid.envs:
        env._gym_env.env.save_for_render(dir="renders")
    tf_env_valid = create_env(start=start, finish=finish)
    tf_env_valid.reset()
    policy_state = agent.policy.get_initial_state(tf_env_valid.batch_size)
    iteration = 0
    window_iter = []
    print(f"\nMoving to a new random window starting at {start}.")

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlabel('Iteration')
ax.set_ylabel('Portfolio Valuation')
ax.set_title('Live Portfolio Valuation Plot')
line_styles = ['-', '--', '-.', ':']
colors = plt.colormaps['tab10']

# Plot each window's valuations as a separate line
for i, window_vals in enumerate(portfolio_valuations):
    final_val = window_vals[-1]
    if final_val > 1000:
        # Highlight lines with final valuation > 1000
        line, = ax.plot(range(len(window_vals)), window_vals, lw=2, label=f"Window {i+1} (highlighted)",
                        linestyle=line_styles[i % len(line_styles)], color='red', alpha=0.75, marker='o', markersize=3)
    else:
        line, = ax.plot(range(len(window_vals)), window_vals, lw=2, label=f"Window {i+1}",
                        linestyle=line_styles[i % len(line_styles)], color=colors(i), alpha=0.75, marker='o', markersize=3)

# Adjust the plot limits and add the legend
ax.relim()
ax.autoscale_view(True, True, True)
ax.legend(loc='upper left', fontsize='small')

# Add gridlines for better readability
ax.grid(True)

plt.show()

actions_taken = [actions_taken[i].action.numpy()[0]
                 for i in range(len(actions_taken))]
actions_plt = []
for action in actions_taken:
    actions_plt.append(1 if positions[action] > 0 else (
        2 if positions[action] < 0 else 0))

value_counts = {
    0: actions_plt.count(0),
    1: actions_plt.count(1),
    2: actions_plt.count(2)
}

# Create a list of labels and values
labels = ['No Position', 'Long Position', 'Short Position']
values = [value_counts[0], value_counts[1], value_counts[2]]

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
ax.set_title('Distribution of Positions')

plt.show()
