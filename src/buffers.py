from model import agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

new_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=10_000_000
)

new_replay_buffer_observer = new_replay_buffer.add_batch


def copy_new_replay_buffer(src_buffer, dst_buffer):
    for i in range(src_buffer.num_frames().numpy()):
        experience, _ = src_buffer.get_next(sample_batch_size=1)
        dst_buffer.add_batch(experience)
        print(f'Iteration: {i}/1_000_000', end='\r')
