from src.preprocessing import preprocess_data
from src.hyperparameter_tuning import tune_hyperparameters
from src.train_models import train_all_models

if __name__ == "__main__":
    X_train , y_train = preprocess_data(".\\datasets\\train.parquet",'train')
    X_val , y_val = preprocess_data(".\\datasets\\val.parquet",'validate')
    # print(len(X_train[0]))
    # print("---")
    # print(len(X_val[0]))
    train_all_models(X_train, y_train , X_val , y_val)
    # tune_hyperparameters(X_train , y_train)
