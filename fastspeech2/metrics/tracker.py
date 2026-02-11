import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Init metric tracker
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.
        """
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        Return average value for a given metric.
        """
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.
        """
        return dict(self._data.average)

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.
        """
        return self._data.total.keys()
