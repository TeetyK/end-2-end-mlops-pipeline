import joblib
import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
def tune_hyperparameters(X_train , y_train):
    
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("Hyperparameter Tuning")

    with mlflow.start_run():
        model = RandomForestClassifier()

        param_dist = {
            'n_estimators':[500],
            'max_depth':[10,None],
            'min_samples_split':[10],
            'min_samples_leaf':[1],
            'max_features':['sqrt']
        }
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist,
            n_iter=5 , cv=5 , scoring='f1_weighted',random_state=9
        )

        random_search.fit(X_train , y_train)

        best_model = random_search.best_estimator_

        mlflow.sklearn.log_model(best_model,"RandomForest_best")
        joblib.dump(best_model,'.\\models\\tuned_models\\best_model.joblib')
        print(f"Best Parameters : {random_search.best_params_}")
