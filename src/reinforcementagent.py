from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.utils.common import function
from tf_agents.environments import tf_py_environment
from tensorflow import keras
from model import q_net, agent
from fe import train, valid, test
from env import tf_env, positions, GymWrapper
from reward import calculate_reward
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tf_agents
import os
import gymnasium as gym
from drivers import collect_driver, train_metrics, new_replay_buffer, collect_data

train.drop(columns=["date_close"], inplace=True)
valid.drop(columns=["date_close"], inplace=True)
test.drop(columns=["date_close"], inplace=True)
print(train.info())
z_cols = [col for col in train.columns if col.startswith('z_')]
print(train[z_cols].info())
print(train[z_cols].head())
print(train.head())

save_history = []

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"TF-Agents version: {tf_agents.__version__}")
tf_env.reset()

agent.initialize()


# checkpoint_dir = './kaggle/working/checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(agent=agent, rb=new_replay_buffer)
#
# checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
#
# if checkpoint_manager.latest_checkpoint:
#     checkpoint.restore('./kaggle/working/checkpoints/ckpt-308').expect_partial()
#     print("Restored checkpoint from {}".format('./kaggle/working/checkpoints/ckpt-308'))
# else:
#     print("No checkpoint found. Starting from scratch.")


# new_checkpoint_dir = './kaggle/working/checkpoints_new'
# new_checkpoint_prefix = os.path.join(new_checkpoint_dir, "ckpt")
# new_checkpoint = tf.train.Checkpoint(agent=agent, rb=new_new_replay_buffer)
#
# new_checkpoint_manager = tf.train.CheckpointManager(new_checkpoint, new_checkpoint_dir, max_to_keep=3)
#
# new_checkpoint_manager.save()

final_time_step, final_policy_state = collect_data()
print(final_time_step)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


def create_logger(exp_version):
    log_file = ("{}.log".format(exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)


VERSION = "001"  # 実験番号
create_logger(VERSION)

logger = get_logger(VERSION)


def log_stuff(train_metrics):
    log_metrics(train_metrics)

    metric_names = ["NumberOfEpisodes", "EnvironmentSteps", "AverageReturnMetric",
                    "AverageEpisodeLengthMetric", "PortfolioValuationMetric"]

    for i, metric in enumerate(train_metrics):
        logger.info(f"{metric_names[i]}: {metric.result().numpy()}")


tf_env.reset()


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


new_checkpoint_dir = './kaggle/working/checkpoints_new'
new_checkpoint_prefix = os.path.join(new_checkpoint_dir, "ckpt")
new_checkpoint = tf.train.Checkpoint(agent=agent, rb=new_replay_buffer)

new_checkpoint_manager = tf.train.CheckpointManager(
    new_checkpoint, new_checkpoint_dir, max_to_keep=3)

if new_checkpoint_manager.latest_checkpoint:
    # new_checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
    print("Restored checkpoint from {}".format(
        new_checkpoint_manager.latest_checkpoint))
else:
    print("No checkpoint found. Starting from scratch.")


# def train_agent(n_iterations):
#     time_step = tf_env.reset()
#     policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
#     iterator = iter(dataset)
#
#     for iteration in range(n_iterations):
#         if time_step.is_last():
#             tf_env.reset()
#         time_step, policy_state = collect_driver.run(time_step, policy_state, maximum_iterations=1000)
#
#         trajectories, buffer_info = next(iterator)
#         train_loss = agent.train(trajectories)
#         print("\r{} F. straty:{:.5f}".format(iteration, time_step.reward.numpy()[0]), end="")
#         if iteration % 10_000 == 0 and iteration != 0:
#             with open('/kaggle/working/iteration.pkl', 'wb') as f:
#                 pickle.dump(iteration+start, f)
#             log_stuff(train_metrics)
#             log = tf_env.envs[0].logging_buffer()
#             portfolio_valuation.append(log['portfolio_valuation'])
#             logger.info("Portfolio valuation: " + str(log['portfolio_valuation']))
#             logger.info("Reward: " + str(log['reward']))
#             new_checkpoint_manager = tf.train.CheckpointManager(new_checkpoint, new_checkpoint_dir, max_to_keep=3)
#             new_checkpoint_manager.save()
#             logger.info(f"Saved checkpoint for iteration {iteration}")
@tf.function
def debug_train_step(experience):
    with tf.GradientTape() as tape:
        loss_info = agent._loss(
            experience,
            td_errors_loss_fn=agent._td_errors_loss_fn,
            gamma=agent._gamma,
            reward_scale_factor=agent._reward_scale_factor,
            weights=None,
            training=True,
        )
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    return loss_info.loss


timestep = tf_env.reset()


# In[ ]:

portfolio_valuation = []


def train_agent_real(n_iterations):
    global reset_counter
    time_step = tf_env.reset()
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)

        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        print("\r{} F. straty:{:.5f}".format(
            iteration, train_loss.loss), end="")
        if iteration % 10_000 == 0:
            log_stuff(train_metrics)
            log = tf_env.envs[0].logging_buffer()
            portfolio_valuation.append(log['portfolio_valuation'])
            logger.info("Portfolio valuation: " +
                        str(log['portfolio_valuation']))


def train_agent(n_epochs):
    iterator = iter(dataset)

    for epoch in range(n_epochs):
        time_step = tf_env.reset()
        policy_state = agent.collect_policy.get_initial_state(
            tf_env.batch_size)
        done = False
        iteration = 0
        while not time_step.is_last():
            time_step, policy_state = collect_driver.run(
                time_step, policy_state)  # maximum_iterations = 1000
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories).loss

            print("\rEpoch: {}, Iter: {} F. straty:{:.5f}, done: {}".format(
                epoch, iteration, train_loss, time_step.is_last()), end="")

            if iteration % 1_000 == 0:
                log_stuff(train_metrics)
                log = tf_env.envs[0].logging_buffer()
                portfolio_valuation.append(log['portfolio_valuation'])
                logger.info("Portfolio valuation: " +
                            str(log['portfolio_valuation']))
            iteration += 1


# log_stuff(train_metrics)
#         log = tf_env.envs[0].logging_buffer()
#         portfolio_valuation.append(log['portfolio_valuation'])
#         logger.info("Portfolio valuation: " + str(log['portfolio_valuation']))
#         logger.info("Reward: " + str(log['reward']))
#         new_checkpoint_manager = tf.train.CheckpointManager(new_checkpoint, new_checkpoint_dir, max_to_keep=3)
#         new_checkpoint_manager.save()
#         logger.info(f"Saved checkpoint for iteration {iteration}")
tf_env.reset()
dataset = new_replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).prefetch(tf.data.experimental.AUTOTUNE)
done = False
observation = tf_env.reset()
iteration = 0
while not done:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    position_index = 1
    timestep = tf_env.step(position_index)
    done = timestep.is_last()
    print(iteration, end="\r")
    iteration += 1
tf_env.step(1)

train_agent(1_000)


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
