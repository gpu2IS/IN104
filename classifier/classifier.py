import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVC

PATH = r'./data/features.csv'

dataset = pd.read_csv(PATH,sep=';')
columns = dataset.columns.tolist() # get the columns

batch_audio = pd.DataFrame(dataset).to_numpy()

features = batch_audio[:, 1:]
y = batch_audio[:, 0]

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.20, random_state=0)

rbs_clf = RobustScaler()
X_train = rbs_clf.fit_transform(X_train)
X_test = rbs_clf.transform(X_test)

model = LinearSVC(C=0.1, max_iter=1e5, tol=1e-4)
model.fit(X_train, y_train)

print("Performances du modèle sur la base de données de test : ", model.score(X_test, y_test))