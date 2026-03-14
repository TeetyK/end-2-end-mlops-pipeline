import pandas as pd
import joblib
from sklearn.metrics import accuracy_score , classification_report

def evaluate_and_predict():
    
    df_test = pd.read_parquet('.\\datasets\\test.parquet', engine='pyarrow')
    X_test = df_test.drop(columns=['Target'])
    y_test = df_test['Target']

    preprocessor = joblib.load('.\\models\\transformers\\preprocessor.joblib')
    le = joblib.load('.\\models\\transformers\\label_encoder.joblib')
    model = joblib.load('.\\models\\tuned_models\\best_model.joblib')

    X_test_processed = preprocessor.transform(X_test)
    y_test_encoded = le.transform(y_test)

    predictions = model.predict(X_test_processed)

    acc = accuracy_score(y_test_encoded, predictions)
    print(f"Test Accuracy : {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_encoded,predictions,target_names=le.classes_))

    predicted_labels = le.inverse_transform(predictions)
    submission_df = pd.DataFrame({
    'Actual_Target':y_test,
    'Predicted_Target': predicted_labels
    })
    submission_df.to_csv('.\\datasets\\submission.csv',index=False)

if __name__ == "__main__":
    evaluate_and_predict()