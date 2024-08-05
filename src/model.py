import tensorflow as tf
from env import tf_env
from tensorflow import keras
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent

lstm_size = (512,)

# Combine the encoding and value networks
q_net = QRnnNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    lstm_size=lstm_size,
)

train_step = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=5_000_000,
    end_learning_rate=0.1
)

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=4_000,
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99,
    gradient_clipping=1.0,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step)
)