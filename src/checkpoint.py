import os
from model import agent
from buffers import new_replay_buffer
import tensorflow as tf


new_checkpoint_dir = '../checkpoints/'
new_checkpoint_prefix = os.path.join(new_checkpoint_dir, "ckpt")
new_checkpoint = tf.train.Checkpoint(agent=agent, rb=new_replay_buffer)

new_checkpoint_manager = tf.train.CheckpointManager(
    new_checkpoint, new_checkpoint_dir, max_to_keep=3)


def load_checkpoint():
    if new_checkpoint_manager.latest_checkpoint:
        new_checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
        print("Restored checkpoint from {}".format(
            new_checkpoint_manager.latest_checkpoint))
    else:
        print("No checkpoint found. Starting from scratch.")
def save_checkpoint():
    new_checkpoint_manager.save()
    print("Saved checkpoint")
