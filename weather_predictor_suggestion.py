import pandas as pd

lufttempData = pd.read_json(path_or_buf="weatherData/lufttemp.json", orient='records')
lufttryckData = pd.read_json(path_or_buf="weatherData/lufttryck.json", orient='records')
relluftfuktData = pd.read_json(path_or_buf="weatherData/relluftfukt.json", orient='records')
siktData = pd.read_json(path_or_buf="weatherData/sikt.json", orient='records')
weatherData = pd.read_json(path_or_buf="weatherData/weather.json", orient='records')
weatherCodesData = pd.read_json(path_or_buf="weatherData/weather_codes.json", orient='records')
vindhastData = pd.read_json(path_or_buf="weatherData/vindhast.json", orient='records')
vindriktData = pd.read_json(path_or_buf="weatherData/vindrikt.json", orient='records')

lufttempData.rename({'value':'airTemp'})

# print(len(lufttempData), lufttempData.head(3))
# print(len(lufttryckData), lufttryckData.head(3))
# print(len(relluftfuktData), relluftfuktData.head(3))
# print(len(siktData), siktData.head(3))
# print(len(weatherData), weatherData.head(3))
# #print(weatherCodesData.head(3))
# print(len(vindhastData), vindhastData.head(3))
# print(len(vindriktData), vindriktData.head(3))

data = pd.DataFrame()#columns=['date', 'airTemp', 'airPressure', 'relAirHumidity', 'sightDistance', 'weather', 'windSpeed', 'windDirection'])

date_new = relluftfuktData.loc[0:1952].date

airTemp_new = pd.Series([])
airPressure_new = pd.Series([])
relAirHumidity_new = pd.Series([])
sightDistance_new = pd.Series([])
weather_new = pd.Series([])
windSpeed_new = pd.Series([])
windDirection_new = pd.Series([])

data.insert(0, 'date', date_new)
# print(len(relluftfuktData), len(data))
#print(relluftfuktData[2944][:])
for i in range(len(data)):
    if data['date'][i] == lufttempData['date'][i]:
        airTemp_new[i] = lufttempData['value'][i]

    if data['date'][i] == lufttryckData['date'][i]:
        airPressure_new[i] = lufttryckData['value'][i]

    if data['date'][i] == relluftfuktData['date'][i]:
        relAirHumidity_new[i] = relluftfuktData['value'][i]

    if data['date'][i] == siktData['date'][i]:
        sightDistance_new[i] = siktData['value'][i]

    if data['date'][i] == weatherData['date'][i]:
        weather_new[i] = weatherData['value'][i]

    if data['date'][i] == vindhastData['date'][i]:
        windSpeed_new[i] = vindhastData['value'][i]

    if data['date'][i] == vindriktData['date'][i]:
        windDirection_new[i] = vindriktData['value'][i]

data.insert(1, 'airTemp', airTemp_new)
data.insert(2, 'airPressure', airPressure_new)
data.insert(3, 'relAirHumidity', relAirHumidity_new)
data.insert(4, 'sightDistance', sightDistance_new)
data.insert(5, 'weather', weather_new)
data.insert(6, 'windSpeed', windSpeed_new)
data.insert(7, 'windDirection', windDirection_new)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(weatherCodesData.head(3))
    print(data.groupby('weather').mean().merge(weatherCodesData, how='left', left_on='weather', right_on='key'))
    pass

weatherOneHot = pd.get_dummies(data=data.weather, prefix='weather')

#print(weatherOneHot)
#print(data)

data = data.merge(weatherOneHot, how='left', left_index=True, right_index=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(data.head())
    pass

data.to_csv("weatherData/weatherTable.csv")
