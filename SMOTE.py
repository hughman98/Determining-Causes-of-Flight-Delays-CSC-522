import pandas as pd
import numpy as np
import statistics


def distance(p1, p2, median):
    """
    A distance function used in SMOTE-NC

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
                sum_of_squares += median ** 2
            else:
                sum_of_squares += (p1[i] - p2[i]) ** 2

    return np.sqrt(sum_of_squares)


def get_knn(data, p1, k):
    """
    KNN Algorithm

    :param data: A dataframe containing all data that needs to be searched
    :param p1: The datapoint we are looking for neighbors for, as an iterable object
    :param k: How many neighbors we should find.
    :return: A list of neighbors.
    """
    standard_deviations = data.std(numeric_only=True)
    median = statistics.median(standard_deviations)

    distances = list()
    for idx, p2 in data.iterrows():
        distances.append((distance(p1, p2, median), idx))
        distances.sort()

    ret = []
    # This starts at 1 so that the value itself is not returned as a neighbor
    for value in distances[1:k + 1]:
        ret.append(data.iloc[value[1]])

    return ret


def smote(data, n):
    """
    smote implements the SMOTE-NC algorithm

    :param data: A pandas dataframe containing a subset of the dataset being smoted.
     This subject should only contain the minority class being expanded.
     All nominal columns should contain only strings, and all numerical columns should have no strings.
    :param n: The number of new data points that should be created.
    :return: A new pandas dataframe containing all artificial data
    """

    # TODO: Finish this


if __name__ == '__main__':
    # Some basic tests to see if the functions working as intended

    data = {
        "calories": [420, 420, 390, 65, 389, 5684],
        "duration": [50, 50, 45, 56, 34, 87],
        "type": ["ice-cream", "ice-cream", "meat", "veg", "veg", "ice-cream"],
        "id": [0, 1, 2, 3, 4, 5]
    }

    df = pd.DataFrame.from_dict(data)

    median = statistics.median(df.std(numeric_only=True))

    assert distance(df.iloc[0], df.iloc[0], median) == 0
    assert distance(df.iloc[0], df.iloc[1], median) != 0

    assert list(get_knn(df, df.iloc[0], 2)[0]) == list(df.iloc[1])
    assert list(get_knn(df, df.iloc[0], 2)[0]) != list(df.iloc[0])
