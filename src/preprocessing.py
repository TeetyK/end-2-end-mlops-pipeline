import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
import pandas as pd

def preprocess_data(path,mode):
    df = pd.read_parquet(path,engine='pyarrow')
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    course_column = ['Course']
    numerical_columns = X.drop(columns=course_column).columns.tolist()

    
    if mode == "train":
        preprocessor = ColumnTransformer(
        transformers=[
                ('num', StandardScaler() , numerical_columns),
                ('course', OneHotEncoder(handle_unknown='ignore'), course_column)
            ],
            remainder='passthrough'
        )

        le = LabelEncoder()

        preprocessor.fit(X)
        le.fit(y)
        joblib.dump(preprocessor,'.\\models\\transformers\\preprocessor.joblib')
        joblib.dump(le,'.\\models\\transformers\\label_encoder.joblib')
    else:
        preprocessor = joblib.load('.\\models\\transformers\\preprocessor.joblib')
        le = joblib.load('.\\models\\transformers\\label_encoder.joblib')
    
    X_processed = preprocessor.transform(X)
    y_encoded = le.transform(y)

    return X_processed , y_encoded

# if __name__ == "__main__":
#     preprocess_data()