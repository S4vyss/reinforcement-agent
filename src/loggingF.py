import numpy as np
import shutil
from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from tf_agents.eval.metric_utils import log_metrics
from metrics import train_metrics
import sys


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


VERSION = "001"
create_logger(VERSION)

logger = get_logger(VERSION)


class MetricsLogger:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.current_values = {name: 0.0 for name in metric_names}

    def update(self, trajectory):
        # Update current values based on the trajectory
        self.current_values["NumberOfEpisodes"] = self._to_scalar(
            trajectory.reward)
        for name, metric in zip(self.metric_names[1:], train_metrics):
            self.current_values[name] = self._to_scalar(metric.result())

        self.update_progress_bar()

    def _to_scalar(self, value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.item()
        return value

    def update_progress_bar(self):
        terminal_width = shutil.get_terminal_size().columns
        bar_width = terminal_width - 2  # Leave space for start/end brackets
        progress_bar = "["

        # Create segments for each metric
        for name in self.metric_names:
            try:
                value = self.current_values[name]
                bar_segment = f"{name[:3]}: {value:.2f}"
                progress_bar += bar_segment.ljust(
                int(bar_width / len(self.metric_names)), " ") + "|"
            except TypeError:
                print(name[:3], value, name)

        progress_bar = progress_bar[:-1] + "]"  # Remove the last '|'

        # Print the progress bar in place
        sys.stdout.write("\r" + progress_bar)
        sys.stdout.flush()


# Usage remains the same
metrics_logger = MetricsLogger([
    "NumberOfEpisodes",
    "EnvironmentSteps",
    "AverageReturnMetric",
    "AverageEpisodeLengthMetric",
    "PortfolioValuationMetric"
])


def log_stuff(trajectory):
    metrics_logger.update(trajectory)
# TODO: Fix the loggingF.py file
