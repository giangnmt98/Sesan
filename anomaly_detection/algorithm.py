import numpy as np


def detect_with_mean_and_std(data, threshold = 3):
    mean = np.mean(data)
    std = np.std(data)
    outliers = [x for x in data if abs(x - mean) > threshold * std]
    return outliers

def detect_with_iqr(data, q1_threshold=25, q3_threshold=75):
    Q1 = np.percentile(data, q1_threshold)
    Q3 = np.percentile(data, q3_threshold)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers

