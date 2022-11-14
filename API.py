import os
from flask import Flask
from flask import request
from flask_cors import CORS
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import keras
import tensorflow
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

##### Questionnaire SVM model
dataDf = pd.read_csv("questionnaire/data.csv")
questionnaire_X = dataDf[['eye_redness', 'eye_pain', 'light_sensitivity', 'blurred_vision', 'floating_spots']]
questionnaire_y = dataDf['uveitis']
questionnaire_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
questionnaire_clf.fit(questionnaire_X, questionnaire_y)

img_width, img_height = 350, 350

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

app = Flask(__name__)
cors = CORS(app)

@app.route('/identifyDisease', methods=['POST'])
def identifyDisease():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights('model_saved.h5')

    graph = tensorflow.get_default_graph()
    f = request.files['file']
    f.save(os.path.join("uploads", "upload.jpg"))
    sess = keras.backend.get_session()
    img = tensorflow.read_file("uploads/upload.jpg")
    img = tensorflow.image.decode_jpeg(img, channels=3)
    img.set_shape([None, None, 3])
    img = tensorflow.image.resize_images(img, (350, 350))
    img = img.eval(session=sess)  # convert to numpy array
    img = np.expand_dims(img, 0)  # make 'batch' of 1
    pred = ''
    with graph.as_default():
        # y = model.predict(X)
        pred = model.predict(img)
    y = ["Normal", "Uveitis"]
    return {
        "pred": y[np.argmax(pred)],
        "accuracy": str(pred[0][np.argmax(pred)])
    }

@app.route("/questionnaire", methods =['POST'])
def questionnaire():
    data = np.array(request.get_json())
    prediction = questionnaire_clf.predict(data)
    print(prediction)
    return {"pred": str(prediction[0])}


if __name__ == '__main__':
    app.run(debug=False, port=5000, host="0.0.0.0")
