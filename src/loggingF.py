from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from metrics import train_metrics, second_metrics
total_iterations = 1400


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


def log_stuff(trajectory):
    metrics_string = str(
        [f"{metric.name}: {metric.result().numpy()}" for metric in train_metrics])
    print(
        f"\rReward: {trajectory.reward.numpy()[0]:+7.4f} Metrics: {metrics_string}", end="\r", flush=True)

# TODO: Fix the loggingF.py file
