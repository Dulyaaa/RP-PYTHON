import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask_cors import CORS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

##### Questionnaire SVM model
questionnaire_dataDf = pd.read_csv("data.csv")
questionnaire_X = questionnaire_dataDf[['eye_redness', 'eye_pain', 'light_sensitivity', 'blurred_vision', 'floating_spots']]
questionnaire_y = questionnaire_dataDf['uveitis']
questionnaire_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
questionnaire_clf.fit(questionnaire_X, questionnaire_y)

app = Flask(__name__)
cors = CORS(app)

@app.route("/questionnaire", methods =['POST'])
def questionnaire():
    data = np.array(request.get_json())
    prediction = questionnaire_clf.predict(data)
    print(prediction)
    return {"pred": str(prediction[0])}

if __name__ == '__main__':
    app.run(debug=False, port=5000, host="0.0.0.0")
