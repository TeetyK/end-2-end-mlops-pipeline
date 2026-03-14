import joblib
from src.models_list import models
from sklearn.metrics import accuracy_score , f1_score
import pandas as pd
from src.hyperparameter_tuning import tune_hyperparameters

def train_all_models(X_train , y_train , X_val , y_val):

    result = {}
    best_score = 0
    best_model_name = ""
    best_model_instance = None
    for model_name , model in models.items():
        print(f"Training {model_name}")
        model.fit(X_train , y_train)

        predictions = model.predict(X_val)
        acc = accuracy_score(y_val , predictions)
        score = f1_score(y_val,predictions , average='weighted')
        result[model_name] = acc
        
        joblib.dump(model,f".\\models\\{model_name}.joblib")
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model_instance = model
    print("Best model : ",best_model_name)
    tune_hyperparameters(best_model_instance,X_train , y_train,best_model_name)
    
    final_result = pd.DataFrame(list(result.items()), columns=['Model', 'F1_Score'])
    final_result.to_csv(".\\datasets\\train_model.csv",index=False)

