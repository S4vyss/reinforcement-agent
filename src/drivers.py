from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from model import agent
from env import tf_env
from buffers import new_replay_buffer_observer, new_replay_buffer
from tf_agents.metrics import tf_metrics

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{} Reward: {}".format(self.counter,
                  self.total, trajectory.reward), end="")


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

collect_driver = DynamicEpisodeDriver(
    tf_env,
    agent.collect_policy,
    observers=[new_replay_buffer_observer,
               ShowProgress(20_000)] + train_metrics,
    num_episodes=1
)
tf_env.step(1)

initial_collect_policy = RandomTFPolicy(
    tf_env.time_step_spec(), tf_env.action_spec())
init_driver = DynamicEpisodeDriver(
    tf_env,
    initial_collect_policy,
    observers=[new_replay_buffer.add_batch, ShowProgress(20_000)],
    num_episodes=1
)

def collect_data():
    final_time_step, final_policy_state = init_driver.run()
    return final_time_step, final_policy_state
