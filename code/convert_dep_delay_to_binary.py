import pandas as pd

df = pd.read_csv('flight_weather_data3.csv', low_memory=False)

# Remove rows with missing values in 'dep_delay' column
df.dropna(subset=['dep_delay'], inplace=True)

# Apply the 'delay_class' classification to 'dep_delay' column
df['dep_delay']=['yes' if x > 10 else 'no' for x in df['dep_delay']]
df.rename(columns={'dep_delay': 'delay_class'}, inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('flight_weather_data_dep_delay_binary.csv', index=False)