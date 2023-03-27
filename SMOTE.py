import pandas as pd
import numpy as np
import statistics

def distance(p1, p2, median):
    """
    An algorithm for distance used in SMOTE-NC

    :param p1: The first datapoint, as an iterable object
    :param p2: The second datapoint, as an iterable object
    :param median: The median SD of numerical categories in the data, to use when
     there are differences in categorical data
    :return: The calculated distance between p1 and p2.
    """
    sum_of_squares = 0
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            if isinstance(p1[i], str):
                sum_of_squares += median**2
            else:
                sum_of_squares += (p1[i] - p2[i])**2

    return np.sqrt(sum_of_squares)


def smote(data, n):
    """
    smote implements the SMOTE-NC algorithm

    :param data: A pandas dataframe containing a subset of the dataset being smoted.
     This subject should only contain the minority class being expanded.
     All nominal columns should contain only strings, and all numerical columns should have no strings.
    :param n: The number of new data points that should be created.
    :return: A new expanded pandas dataframe containing new artificial data
    """

    standard_deviations = data.std(numeric_only=True)
    median = statistics.median(standard_deviations)

    # TODO: Finish this


if __name__ == '__main__':
    # A basic test to see if the smote algorithm is working as intended

    data = {
        "calories": [420, 380, 390, 65, 389, 5684],
        "duration": [50, 40, 45, 56, 34, 87],
        "type": ["ice-cream", "ice-cream", "meat", "veg", "veg", "ice-cream"],
        "id": [0, 1, 2, 3, 4, 5]
    }

    df = pd.DataFrame.from_dict(data)

    median = statistics.median(df.std(numeric_only=True))

    assert distance(df.iloc[0], df.iloc[0], median) == 0
    assert distance(df.iloc[0], df.iloc[1], median) != 0
