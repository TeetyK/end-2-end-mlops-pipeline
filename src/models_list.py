from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVC':SVC(),
    'DecisionTree':DecisionTreeClassifier(),
    'GradientBoosting':GradientBoostingClassifier(),
    'AdaBoost':AdaBoostClassifier(),
    'KNN':KNeighborsClassifier(),
    'GaussianNB': GaussianNB()
}
