import matplotlib.pyplot as plot
import numpy as np


def plot_loss(loss_dict, path):
    """
    :param loss_dict: {"train":loss_array, "dev":loss_array}
    :return: plot
    """
    for name in loss_dict.keys():
        loss_array = loss_dict[name]
        plot.plot(range(1, len(loss_array) + 1),
                  loss_array, label=name)
        plot.title(name+"_loss")
    plot.xlabel("epoch")
    plot.tight_layout()
    plot.legend()
    plot.savefig(path)

