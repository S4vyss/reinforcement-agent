import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from loggingF import log_stuff
from model import agent
from env import tf_env
from buffers import new_replay_buffer_observer, new_replay_buffer
from metrics import train_metrics
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils


class ShowProgress:
    def __init__(self, total):
        self.counter = tf.Variable(0, dtype=tf.int32)
        self.total = total

    @tf.function
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter.assign_add(1)
        if self.counter % 100 == 0:
            print("\r{}/{} Reward: {}".format(self.counter.numpy(),
                  self.total, trajectory.reward.numpy()), end="")


collect_driver = DynamicEpisodeDriver(
    tf_env,
    agent.collect_policy,
    observers=[new_replay_buffer_observer, log_stuff] + train_metrics,
    num_episodes=1
)

test_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[new_replay_buffer_observer, log_stuff] + train_metrics,
    num_steps=1
)

initial_collect_policy = RandomTFPolicy(
    tf_env.time_step_spec(), tf_env.action_spec())
init_driver = DynamicEpisodeDriver(
    tf_env,
    initial_collect_policy,
    observers=[new_replay_buffer.add_batch, log_stuff] + train_metrics,
    num_episodes=1
)

def collect_data():
    print("Start collecting data")
    final_time_step, final_policy_state = init_driver.run()

    return final_time_step, final_policy_state


def collect_step(environment, policy_state, policy, buffer, observers):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)
    
    for observer in observers:
        observer(traj)
    
    return action_step.state


def collect_data_test(env, policy, policy_state, buffer, steps):
    for _ in range(steps):
        policy_state = collect_step(env, policy_state, policy, buffer, [log_stuff, *train_metrics])
    return policy_state
