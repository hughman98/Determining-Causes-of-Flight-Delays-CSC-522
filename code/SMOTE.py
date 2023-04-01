import pandas as pd
import numpy as np
import statistics
import random


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


def smote(mc, n, k):
    """
    smote implements the SMOTE-NC algorithm, details of which can be found here: https://arxiv.org/pdf/1106.1813.pdf

    :param mc: A pandas dataframe containing a subset of the dataset being smoted.
     This subject should only contain the minority class being expanded.
     All nominal columns should contain only strings, and all numerical columns should have no strings.
    :param n: The number of new data points that should be created. It cannot be larger than the length of data
    :param k: The number of nearest neighbors to use during synthesis.
    :return: A new pandas dataframe containing all artificial data
    """

    # Randomize order
    mc = mc.sample(frac=1)

    dict = {}
    for col in mc.columns:
        dict[col] = []

    count = 0
    while count <= n:
        for idx, data_point in mc.iterrows():
            count += 1
            if count > n:
                break

            neighbors = get_knn(mc, data_point, k)
            picked = neighbors[random.randint(0, k - 1)]

            for col in mc.columns:
                if isinstance(data_point[col], str):  # nominal columns use mode of neighbors
                    neighbor_values = []
                    for neighbor in neighbors:
                        neighbor_values.append(neighbor[col])
                    dict[col].append(statistics.mode(neighbor_values))

                else:  # numeric columns use the mean with our randomly picked neighbor
                    dict[col].append(statistics.mean((data_point[col], picked[col])))

    return pd.DataFrame.from_dict(dict)


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

    print(smote(df, 3, 5))
