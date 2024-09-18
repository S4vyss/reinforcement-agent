import os
import tensorflow as tf


class CheckpointManager:
    def __init__(self, agent, replay_buffer, checkpoint_dir='/home/s4vyss/Projekty/reinforcement-agent/checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(agent=agent, rb=replay_buffer)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def load_checkpoint(self):
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored checkpoint from {self.manager.latest_checkpoint}")
            return True
        else:
            print("No checkpoint found. Starting from scratch.")
            return False

    def save_checkpoint(self):
        self.manager.save()
        print("Saved checkpoint")


checkpoint_manager = None


def initialize_checkpoint_manager(agent, replay_buffer):
    global checkpoint_manager
    checkpoint_manager = CheckpointManager(agent, replay_buffer)


def load_checkpoint():
    if checkpoint_manager:
        loaded = checkpoint_manager.load_checkpoint()
        if loaded:
            return True
        else:
            return False
    else:
        print(
            "Checkpoint manager not initialized. Call initialize_checkpoint_manager first.")


def save_checkpoint():
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint()
    else:
        print(
            "Checkpoint manager not initialized. Call initialize_checkpoint_manager first.")
