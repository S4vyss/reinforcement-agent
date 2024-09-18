from metrics import train_metrics
import logging

logging.basicConfig(filename="logs/logs.log", filemode="w",
                    format="%(message)s")

total_iterations = 1400


def log_stuff(trajectory):
    # logging.info(trajectory.action)
    metrics_string = str(
        [f"{metric.name}: {metric.result().numpy()}" for metric in train_metrics])
    print(
        f"\rReward: {trajectory.reward.numpy()[0]:+7.4f} Metrics: {metrics_string}", end="\r", flush=True)

# TODO: Fix the loggingF.py file
