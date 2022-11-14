import numpy
import pandas as pd
from datetime import datetime

f = "%Y-%m-%d"

dataDf = pd.read_csv("data.csv")

startDate = datetime.strptime(str(dataDf.iloc[0]["date"]), f)

dayIndexes = []

for index, row in dataDf.iterrows():
    dayIndexes.append((datetime.strptime(row["date"], f) - startDate).days)

dataDf["dayIndexes"] = dayIndexes

X = dataDf[['dayIndexes']]
y_eye_redness = dataDf['eye_redness']
y_eye_pain = dataDf['eye_pain']
y_light_sensitivity = dataDf['light_sensitivity']
y_blurred_vision = dataDf['blurred_vision']
y_floating_spots = dataDf['floating_spots']

from sklearn.linear_model import LinearRegression

eye_redness_reg = LinearRegression().fit(X, y_eye_redness)
eye_pain_reg = LinearRegression().fit(X, y_eye_pain)
light_sensitivity_reg = LinearRegression().fit(X, y_light_sensitivity)
blurred_vision_reg = LinearRegression().fit(X, y_blurred_vision)
floating_spots_reg = LinearRegression().fit(X, y_floating_spots)

prediction_date = datetime.strptime("2022-02-01", f)
days_from_start_date = (prediction_date - startDate).days

print("Eye redness by "+str(prediction_date)+": "+str(eye_redness_reg.predict(numpy.array([[days_from_start_date]]))[0]))
print("Eye pain by "+str(prediction_date)+": "+str(eye_pain_reg.predict(numpy.array([[days_from_start_date]]))[0]))
print("Light sensitivity by "+str(prediction_date)+": "+str(light_sensitivity_reg.predict(numpy.array([[days_from_start_date]]))[0]))
print("Blurred vision by "+str(prediction_date)+": "+str(blurred_vision_reg.predict(numpy.array([[days_from_start_date]]))[0]))
print("Floating spots by "+str(prediction_date)+": "+str(floating_spots_reg.predict(numpy.array([[days_from_start_date]]))[0]))
