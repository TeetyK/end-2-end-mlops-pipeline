# Dataset and Project
learning concept from https://www.geeksforgeeks.org/machine-learning/end-to-end-mlops-pipeline-a-comprehensive-project/

# Result Test model
```bash
Test Accuracy : 0.7695

Classification Report:
               precision    recall  f1-score   support

     Dropout       0.83      0.77      0.80       316
    Enrolled       0.54      0.30      0.38       151
    Graduate       0.77      0.94      0.85       418

    accuracy                           0.77       885
   macro avg       0.72      0.67      0.68       885
weighted avg       0.75      0.77      0.75       885
```

# Running on Project
running mlflow
```bash
uv run mlflow ui --port 5000
```
running train_model
```bash
uv run main.py
```
deployment with fastapi
```bash
uv run app.py
```
test api 
```bash
uv run test_api.py
```