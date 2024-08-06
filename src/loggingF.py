from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from tf_agents.eval.metric_utils import log_metrics
from metrics import train_metrics


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
    log_metrics(train_metrics, prefix="\r")

    metric_names = ["NumberOfEpisodes", "EnvironmentSteps", "AverageReturnMetric",
                    "AverageEpisodeLengthMetric", "PortfolioValuationMetric"]

    log_message = f"Reward: {trajectory.reward.numpy()}\n" + " | ".join(
        [f"{name}: {metric.result().numpy()}\n" for name, metric in zip(metric_names, train_metrics)])

    print(log_message, end="\r")
# TODO: Fix the loggingF.py file