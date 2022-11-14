import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

dataDf = pd.read_csv("MOCK_DATA.csv")

print(dataDf.head())

sns.pairplot(dataDf)
plt.show()
sns.heatmap(dataDf.corr(), annot=True)
plt.show()

corr = dataDf.corr(method='pearson')
print(corr)

from sklearn.model_selection import train_test_split

X = dataDf[['distance', 'readable', 'row']]
y = dataDf['acceptable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

######### K Nearest classifier ################
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

score = neigh.score(X_test, y_test, sample_weight=None)
print("K Nearest score: ", score)

print(neigh.predict([[7,0,7]]))

######### Descision tree ################
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test, sample_weight=None)
print("Descision tree score: ", score)

######### Gaussian Naive Bayes ################
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

score = gnb.score(X_test, y_test, sample_weight=None)
print("Gaussian Naive Bayes score: ", score)
