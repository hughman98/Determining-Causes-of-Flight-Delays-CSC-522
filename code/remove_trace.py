import pandas as pd


def remove_trace(df):
    cols = ['Precipitation', 'New Snow', 'Snow Depth']

    for col in cols:
        df[col + ' Binary'] = ['yes' if i == 'T' or float(i) != 0 else 'no' for i in df[col]]
        df[col] = [0 if i == 'T' else i for i in df[col]]

    return df


if __name__ == '__main__':
    # Some basic tests to see if the function is working as intended
    df = pd.read_csv('../data/flight_weather_data3.csv', low_memory=False)
    new_df = remove_trace(df)
    new_df.to_csv("../data/deleteme.csv", index=False)
