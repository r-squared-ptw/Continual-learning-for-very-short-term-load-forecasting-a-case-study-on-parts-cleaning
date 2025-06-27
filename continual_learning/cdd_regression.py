import typing

import numpy as np
import pandas as pd
from river.drift import ADWIN as adwin
from river.drift import KSWIN as kswin
from river.drift.binary import EDDM as eddm
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error


class ConceptDriftDetector:
    """
    Basic Concept Drift Detector using the average error and standard deviation to detect Concept Drifts.
    """

    def __init__(self, warning_level: float, alert_level: float, window_size: int, window_overlap: int = 0):
        """
        :param warning_level: Confidence interval for when the detector should throw a warning.
        :param alert_level: Confidence interval for when the detector should throw an alert. Needs to be higher than warning_level.
        :param window_size: The number of datapoints that shall be used for each evaluation.
        :param window_overlap: The number of datapoints that overlap with the last window used for evaluation. Default: 0.
        """
        self.warning_level = warning_level
        self.alert_level = alert_level
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.warning_level_zscore = norm.ppf(warning_level)
        self.alert_level_zscore = norm.ppf(alert_level)
        self.error_min = None
        self.standard_deviation_min = None
        self.warning_level_reached = False
        self.alert_level_reached = False
        self.iteration = 1
        self.y_true = pd.DataFrame()
        self.y_pred = pd.DataFrame()

        assert window_size > window_overlap

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Collects data until window_size is reached. If enough data is collected concept drift detection is done by comparing the previous average error of the window and the current average error with regards to the standard deviation.

        :param y_true: a single or multiple datapoints containing the true targets.
        :param y_pred: a single or multiple datapoints containing the predictions for the true targets (y_true).
        """
        # add new data
        self.y_true = pd.concat([self.y_true, y_true])
        self.y_pred = pd.concat([self.y_pred, y_pred])

        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        # check if window size is reached
        if len(self.y_true) >= self.window_size:
            # compute error and standard deviation for model output
            error = np.sqrt(((self.y_true - self.y_pred) / self.y_true).abs()).mean(axis=1)
            standard_deviation = error.std()
            error = np.mean(error)
            self.iteration += 1

            # test for concept drift
            if self.error_min == None and self.standard_deviation_min == None:
                self.error_min = error
                self.standard_deviation_min = standard_deviation
            else:
                if (
                    error + standard_deviation
                    >= self.error_min + self.standard_deviation_min * self.warning_level_zscore
                ):
                    self.warning_level_reached = True
                    print("Concept Drift Detection reached warning level.")
                else:
                    self.warning_level_reached = False

                if error + standard_deviation >= self.error_min + self.standard_deviation_min * self.alert_level_zscore:
                    self.alert_level_reached = True
                    print("Concept Drift Detection reached alert level.")
                else:
                    self.alert_level_reached = False

            if self.warning_level_reached is False and self.alert_level_reached is False:
                print("No Concept Drift detected.")
                self.error_min = error
                self.standard_deviation_min = standard_deviation

            # reset window
            if self.window_overlap > 0:
                self.y_true = self.y_true[-self.window_overlap :]
                self.y_pred = self.y_pred[-self.window_overlap :]
            else:
                self.y_true = pd.DataFrame()
                self.y_pred = pd.DataFrame()

        else:
            self.warning_level_reached = False
            self.alert_level_reached = False
            print("Not enough data collected for specified window size.")


class PageHinkleyConceptDriftDetector:
    """
    Concept Drift Detector based on Page-Hinkley tests. It will detect a concept drift if the observed mean at some instant is greater then a threshold value lambda.
    """

    def __init__(
        self,
        warning_level: int = 300,
        alert_level: int = 500,
        min_instances: int = 96,
        delta: float = 0.005,
        alpha: float = 1 - 0.0001,
        metric=mean_absolute_error,
    ):
        """
        :param warning_level: The treshold that needs to be crossed to throw a warning.
        :param alert_level: The treshold that needs to be crossed to throw an alert. Needs to be higher than the warning_level.
        :param min_instances: The number of samples that are collected for reference for detecting concept drifts.
        :param delta: The delta factor of the Page-Hinkley test.
        :param alpha: The forgetting factor, which weighs past values and the mean.
        :param metric: The metric that should be used for concept drift detection.
        """
        self.min_instances = min_instances
        self.delta = delta
        self.alert_level = alert_level
        self.warning_level = warning_level
        self.alpha = alpha
        self.metric = metric
        self.error_mean = 0
        self.sample_count = 0
        self.sum = 0
        self.warning_level_reached = False
        self.alert_level_reached = False

        assert self.alert_level > self.warning_level

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Collects data until min_instances is reached. If enough data is collected concept drifts are detected via the Page-Hinkley test.

        :param y_true: a single datapoint containing the true targets.
        :param y_pred: a single datapoint containing the predictions for the true targets (y_true).
        """
        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        # compute error and mean error
        error = self.metric(y_true, y_pred)
        if self.sample_count > 0:
            self.error_mean = self.error_mean + (error - self.error_mean) / float(self.sample_count)
        else:
            self.error_mean = error
        self.sum = max(0.0, self.alpha * self.sum + (error - self.error_mean - self.delta))

        self.sample_count += 1

        # test for concept drift
        if self.sample_count < self.min_instances:
            print("Not enough data collected for specified min_instances.")
        else:
            if self.sum > self.alert_level:
                self.alert_level_reached = True
                self.warning_level_reached = True
                print("Concept Drift Detection reached alert level.")
            else:
                if self.sum > self.warning_level:
                    self.alert_level_reached = False
                    self.warning_level_reached = True
                    print("Concept Drift Detection reached warning level.")
                else:
                    self.alert_level_reached = False
                    self.warning_level_reached = False
                    print("No Concept Drift detected.")


class ADWIN:
    """
    Concept Drift Detector using the Adaptive Windowing method.
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        metric=mean_absolute_error,
    ):
        """
        :param delta: The significance value.
        :param clock: The interval in which adwin should test for concept drifts. 1 means every data point.
        :param max_buckets: The maximum number of buckts that ADWIN keeps before merging the buckets.
        :param min_window_length: The minimum length of the subwindows.
        :param grace_period: The number of datapoints that come in before ADWIN starts concept drift detection.
        :param metric: The metric that should be used for concept drift detection.
        """
        self.detector = adwin(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        self.metric = metric
        self.alert_level_reached = False
        self.warning_level_reached = None

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Tests for concept drifts via Adaptive Windowing approach.

        :param y_true: a single datapoint containing the true targets.
        :param y_pred: a single datapoint containing the predictions for the true targets (y_true).
        """
        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        error = self.metric(y_true, y_pred)
        self.detector.update(error)

        if self.detector.drift_detected:
            self.alert_level_reached = True
            print("Concept Drift detected.")
        else:
            self.alert_level_reached = False
            print("No Concept Drift detected.")


class KSWIN:
    """
    Kolmogorov-Smirnov Windowing method for concept drift detection.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int or None = None,
        window: typing.Iterable or None = None,
        metric=mean_absolute_error,
    ):
        """
        :param alpha: The probability for the test statistic of the Kolmogorov-Smirnov-Test.
        :param window_size: The number of datapoints contained in the sliding window.
        :param stat_size: The number of datapoints contained in the statistic window.
        :param seed: Random seed for reproducibility.
        :param metric: The metric that should be used for concept drift detection.
        """
        self.detector = kswin(alpha=alpha, window_size=window_size, stat_size=stat_size, seed=seed, window=window)
        self.metric = metric
        self.alert_level_reached = False
        self.warning_level_reached = None

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Tests for concept drifts via the Kolmogorov-Smirnov-Test.

        :param y_true: a single datapoint containing the true targets.
        :param y_pred: a single datapoint containing the predictions for the true targets (y_true).
        """
        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        error = self.metric(y_true, y_pred)
        self.detector.update(error)

        if self.detector.drift_detected:
            self.alert_level_reached = True
            print("Concept Drift detected.")
        else:
            self.alert_level_reached = False
            print("No Concept Drift detected.")


class EWMAConceptDriftDetector:
    """
    Concept Drift Detector using Exponentially Weighted Moving Average charts.
    """

    def __init__(
        self,
        lamda: float = 0.1,
        L: int = 100,
        min_samples: int = 1000,
        initial_mean: float = None,
        initial_standard_deviation: float = None,
        metric=mean_absolute_error,
    ):
        """
        :param lamda: The smoothing factor. It controls how much recent errors affect the EWMA value.
        :param L: The multiplier for the standard deviation determining the UCL and LCL of the control chart.
        :param min_samples: The minimum number of samples required before starting concept drift detection.
        :param initial_mean: The initial mean value for the EWMA. If None, the mean will be calculated based on the initial samples. Provide an initial mean if you have prior knowledge of the expected error mean.
        :param initial_standard_deviation: The standard deviation of of the errors of the predicted values. If None, the standard deviation will be calculated based on the initial samples. Provide the initial standard deviation if you have prior knowledge of the expected standard deviation.
        :param metric: The metric that should be used for concept drift detection.
        """
        self.lamda = lamda
        self.sigma = initial_standard_deviation
        self.L = L
        self.min_samples = min_samples
        self.starting_samples = []
        self.mean = initial_mean
        self.ewma_value = self.mean
        self.metric = metric
        self.alert_level_reached = False
        self.warning_level_reached = None
        self.data_counter = 0

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Tests for concept drifts via Exponentially Weighted Moving Average charts.

        :param y_true: a single datapoint containing the true targets.
        :param y_pred: a single datapoint containing the predictions for the true targets (y_true).
        """
        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        if self.ewma_value is None:
            # collect errors of datapoints
            error = self.metric(y_true, y_pred)
            self.starting_samples.append(error)
            if self.min_samples <= len(self.starting_samples):
                # compute ewma value
                self.mean = np.mean(self.starting_samples)
                self.ewma_value = self.mean

                if self.sigma is None:
                    # compute standard deviation
                    mean = sum(self.starting_samples) / len(self.starting_samples)
                    variance = sum([((x - mean) ** 2) for x in self.starting_samples]) / len(self.starting_samples)
                    self.sigma = variance**0.5
            else:
                print("Not enough data collected for specified min_samples.")
        else:
            # update EWMA value
            error = self.metric(y_true, y_pred)
            self.ewma_value = self.lamda * error + (1 - self.lamda) * self.ewma_value

            # testing for concept drift
            self.data_counter = self.data_counter + 1
            ucl = self.mean + self.L * self.sigma * np.sqrt(
                (self.lamda / (2 - self.lamda)) * (1 - (1 - self.lamda) ** self.data_counter)
            )
            lcl = self.mean - self.L * self.sigma * np.sqrt(
                (self.lamda / (2 - self.lamda)) * (1 - (1 - self.lamda) ** self.data_counter)
            )

            if self.ewma_value > ucl or self.ewma_value < lcl:
                self.alert_level_reached = True
                print("Concept Drift detected.")
            else:
                self.alert_level_reached = False
                print("No Concept Drift detected.")


class EDDMConceptDriftDetector:
    """
    Early Drift Detection Method for regression problems. Regression problem is mapped into a binary classification by testing for a maximum relative error. If the relative error for a smaple exceeds the boundry it is labeld als falsely "classified".
    """

    def __init__(
        self,
        warm_start: int = 30,
        warning_level: float = 0.95,
        alert_level: float = 0.9,
        tolerance: float = 0.1,
    ):
        """
        :param warm_start: The number of initial samples to be used for warm-starting the EDDM detector. During this warm-up period, no drift detection is performed.
        :param warning_level: Threshold for triggering a warning. (0 < warning_level < 1). The smaller the more conservative.
        :param alert_level: Threshold for triggering an alert. (0 < alert_level < 1). The smaller the more conservative.
        :param tolerance: The tolerance threshold for relative error.
        """
        if warning_level < alert_level:
            raise ValueError("'warning_level' must be greater or equal to 'alert_level'.")

        self.detector = eddm(warm_start=warm_start, alpha=warning_level, beta=alert_level)
        self.tolerance = tolerance
        self.alert_level_reached = False
        self.warning_level_reached = False

    def test_for_cd(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Tests for concept drifts via Exponentially Weighted Moving Average charts.

        :param y_true: a single datapoint containing the true targets.
        :param y_pred: a single datapoint containing the predictions for the true targets (y_true).
        """
        assert len(self.y_pred) == len(self.y_true)

        if not (self.y_true.index == self.y_pred.index).all():
            print(
                "DatetimeIndexes of groundtruth and prediction do not match up. The algorithm will proceed as if the indexes are correct. If this occurs regularly, verfify your data collection."
            )
            self.y_true.index = self.y_pred.index

        error = np.mean(((y_true - y_pred) / y_true).abs())
        if error > self.tolerance:
            self.detector.update(1)
        else:
            self.detector.update(0)

        if self.detector.drift_detected:
            self.warning_level_reached = True
            self.alert_level_reached = True
            print("Concept Drift Detection reached alert level.")
        else:
            if self.detector.warning_detected:
                self.warning_level_reached = True
                self.alert_level_reached = False
                print("Concept Drift Detection reached warning level.")
            else:
                self.warning_level_reached = False
                self.alert_level_reached = False
                print("No Concept Drift detected.")
