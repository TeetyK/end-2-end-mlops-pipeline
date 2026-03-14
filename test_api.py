import pandas as pd
import requests

def test_our_api():
    df_test = pd.read_parquet('.\\datasets\\test.parquet', engine='pyarrow')
    
    student_record = df_test.iloc[0].copy()    
    if 'Target' in student_record:
        student_record.drop(labels=['Target'], inplace=True)
    payload = {
        "data": student_record.to_dict()
    }
    url = "http://127.0.0.1:8000/predict"
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"โมเดลทำนายว่า: {result['prediction']}")
        
        actual_target = df_test.iloc[0]['Target']
        print(f"ความจริงแล้วนักศึกษาคนนี้: {actual_target}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_our_api()