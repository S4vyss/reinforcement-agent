from tf_agents.utils.common import function
from model import agent
from fe import train, valid, test
from env import tf_env
import tensorflow as tf
from drivers import collect_driver, new_replay_buffer, collect_data
from loggingF import logger, log_stuff
from metrics import train_metrics

train.drop(columns=["date_close"], inplace=True)
valid.drop(columns=["date_close"], inplace=True)
test.drop(columns=["date_close"], inplace=True)

agent.initialize()

# Initial data collection
final_time_step, final_policy_state = collect_data()
print(final_time_step)

# Wrapping the driver and agent functions
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

portfolio_valuation = []
save_history = []


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


def train_agent_real(n_iterations):
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
        while not time_step.is_last():
            time_step, policy_state = collect_driver.run(
                time_step, policy_state)  # maximum_iterations = 1000
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories).loss
            log = tf_env.envs[0].logging_buffer()
            portfolio_valuation.append(log['portfolio_valuation'])

            log_stuff(train_metrics, epoch, train_loss,
                      log['portfolio_valuation'])
        logger.info(f"Epoch: {epoch}\n Train loss: {train_loss}\n Valuation: {portfolio_valuation}\n")


# log_stuff(train_metrics)
#         log = tf_env.envs[0].logging_buffer()
#         portfolio_valuation.append(log['portfolio_valuation'])
#         logger.info("Portfolio valuation: " + str(log['portfolio_valuation']))
#         logger.info("Reward: " + str(log['reward']))
#         new_checkpoint_manager = tf.train.CheckpointManager(new_checkpoint, new_checkpoint_dir, max_to_keep=3)
#         new_checkpoint_manager.save()
#         logger.info(f"Saved checkpoint for iteration {iteration}")

dataset = new_replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).prefetch(tf.data.experimental.AUTOTUNE)

train_agent(1_000)
