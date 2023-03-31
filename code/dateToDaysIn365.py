import pandas as pd

# raw_data = pd.read_csv('flight_data.csv')
# df = pd.DataFrame(raw_data)

def convertDateToDaysIn365(df):
  monthStartDay=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334] # starting day of each month
  df['days_in_365'] = 0

  for index, row in df.iterrows():
      days = monthStartDay[row['month']-1] + row['day']
      df.at[index, 'days_in_365'] = days

  return df
