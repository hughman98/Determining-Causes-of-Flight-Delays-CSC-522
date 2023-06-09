import pandas as pd
import os

# Get path of the file directory where merge.py exists
cur_dir = os.path.dirname(__file__)

flight_data = pd.read_csv(cur_dir + '/../data/flight_data.csv')
ewr_weather_data = pd.read_csv(cur_dir + '/../data/EWR_Weather.csv')
jfk_weather_data = pd.read_csv(cur_dir + '/../data/JFK_Weather.csv')
lga_weather_data = pd.read_csv(cur_dir + '/../data/LGA_Weather.csv')

flight_data['date'] = pd.to_datetime(flight_data[['year', 'month', 'day']])

ewr_weather_data['Date'] = pd.to_datetime(ewr_weather_data['Date'])
jfk_weather_data['Date'] = pd.to_datetime(jfk_weather_data['Date'])
lga_weather_data['Date'] = pd.to_datetime(lga_weather_data['Date'])


weather_data_dict = {
    'EWR': ewr_weather_data,
    'JFK': jfk_weather_data,
    'LGA': lga_weather_data
}
print(flight_data['date'])
print(ewr_weather_data['Date'])

for index, row in flight_data.iterrows():

    origin_airport = row['origin']

    matching_weather_data = weather_data_dict[origin_airport]
    matching_weather_data = matching_weather_data[(matching_weather_data['Date'] == row['date'])]


    for col in matching_weather_data.columns:
        if col != 'Date':
            flight_data.loc[index, col] = matching_weather_data.iloc[0][col]

flight_data.to_csv(cur_dir + '/../data/flight_weather_data.csv', index=False)