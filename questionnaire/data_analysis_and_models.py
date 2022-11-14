import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

dataDf = pd.read_csv("data.csv")

print(dataDf.head())

sns.pairplot(dataDf)
plt.show()
sns.heatmap(dataDf.corr(), annot=True)
plt.show()

corr = dataDf.corr(method='pearson')
print(corr)

from sklearn.model_selection import train_test_split

X = dataDf[['eye_redness', 'eye_pain', 'light_sensitivity', 'blurred_vision', 'floating_spots']]
y = dataDf['uveitis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

# LOGISTIC REGRESSION
clf = LogisticRegression(random_state=0, max_iter=2000, solver='lbfgs').fit(X_train, y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
print("Logistic regression score: ", score)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

# SVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
score = clf.score(X, y, sample_weight=None)
print("SVM score: ", score)
predictions = clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()

