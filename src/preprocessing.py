import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
import pandas as pd

def preprocess_data():
    df = pd.read_parquet(".\\datasets\\train.parquet",engine='pyarrow')
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    course_column = ['Course']
    numerical_columns = X.drop(columns=course_column).columns.tolist()

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

if __name__ == "__main__":
    preprocess_data()