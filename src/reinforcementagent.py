from tf_agents.utils.common import function
import tf_agents
from fe import train, valid, test
from env import tf_env
import tensorflow as tf
from drivers import collect_driver, collect_data, test_driver, collect_data_test
import warnings
from model import agent
from buffers import new_replay_buffer, replay_buffer
from keras.optimizers.schedules import PolynomialDecay
from checkpoint import load_checkpoint, save_checkpoint, initialize_checkpoint_manager

warnings.filterwarnings("ignore")
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)

train.drop(columns=["date_close"], inplace=True)
valid.drop(columns=["date_close"], inplace=True)
test.drop(columns=["date_close"], inplace=True)

agent.initialize()
initialize_checkpoint_manager(agent, new_replay_buffer)
loaded = False  # load_checkpoint()

random_policy = tf_agents.policies.random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                                   tf_env.action_spec())


# Initial data collection
if not loaded:
    collect_data_test(tf_env, random_policy, random_policy.get_initial_state(
        tf_env.batch_size), replay_buffer, 1400)

# Wrapping the driver and agent functions
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

portfolio_valuation = []


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


beta_PER_fn = PolynomialDecay(
    initial_learning_rate=0.00,
    end_learning_rate=1.00,
    decay_steps=1000*1400
)


def test_train(n_epochs):
    iterator = iter(test_dataset)

    for epoch in range(n_epochs):
        collect_policy = agent.collect_policy
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)
        for i in range(1400):
            policy_state = collect_data_test(env=tf_env, policy_state=policy_state, policy=collect_policy,
                                             buffer=replay_buffer, steps=1)
            trajectories, buffer_info = next(iterator)
            learning_weights = (
                1/(tf.clip_by_value(buffer_info.probabilities, 0.000001, 1.0)*64))**beta_PER_fn(i)
            train_loss, extra = agent.train(
                experience=trajectories, weights=learning_weights)
            replay_buffer.update_batch(buffer_info.ids, extra.td_loss)
        log = tf_env.envs[0].logging_buffer()
        portfolio_valuation.append(log['portfolio_valuation'])
        print(
            f"Epoch: {epoch}\n Train loss: {train_loss}\n Valuation: {portfolio_valuation}\n")
        save_checkpoint()


def train_agent(n_epochs):
    iterator = iter(dataset)

    for epoch in range(n_epochs):
        time_step = tf_env.reset()
        policy_state = agent.collect_policy.get_initial_state(
            tf_env.batch_size)
        time_step, policy_state = collect_driver.run(
            time_step, policy_state)  # maximum_iterations = 1000
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories).loss
        if tf.math.is_nan(train_loss) or train_loss > 1000:
            print("Loss became NaN or too large. Stopping training.")
            break
        log = tf_env.envs[0].logging_buffer()
        portfolio_valuation.append(log['portfolio_valuation'])
        print(
            f"Epoch: {epoch}\n Train loss: {train_loss}\n Valuation: {portfolio_valuation}\n")
        save_checkpoint()


dataset = new_replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

test_train(1000)
