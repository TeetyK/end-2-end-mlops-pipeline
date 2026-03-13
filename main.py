from src.preprocessing import preprocess_data
from src.hyperparameter_tuning import tune_hyperparameters

if __name__ == "__main__":
    X_train , y_train = preprocess_data()

    tune_hyperparameters(X_train , y_train)
