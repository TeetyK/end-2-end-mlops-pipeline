import pandas as pd
from sklearn.model_selection import train_test_split
def extract_data():
    df = pd.read_csv(".\\datasets\\data.csv")
    train_df , test_df = train_test_split(df , test_size=.2,random_state=42)
    train_df , val_df = train_test_split(train_df, test_size=.1,random_state=42)

    train_df.to_parquet('.\\datasets\\train.parquet', engine='pyarrow',index=False)
    test_df.to_parquet('.\\datasets\\test.parquet', engine='pyarrow',index=False)
    val_df.to_parquet('.\\datasets\\val.parquet', engine='pyarrow',index=False)

if __name__ == "__main__":
    extract_data()