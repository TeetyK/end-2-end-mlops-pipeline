from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Student Academic Risk API",
    description="API สำหรับทำนายความเสี่ยงทางการศึกษาของนักศึกษา (MLOps Pipeline)"
)

try:
    preprocessor = joblib.load('.\\models\\transformers\\preprocessor.joblib')
    label_encoder = joblib.load('.\\models\\transformers\\label_encoder.joblib')
    model = joblib.load('.\\models\\tuned_models\\best_model.joblib')
except Exception as e:
    print(f"Error loading models : {e}")

class StudentData(BaseModel):
    data: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to Student Success Predictor API. Go to /docs to test the API."}

@app.post("/predict")
def predict_status(student: StudentData):
    try:
        input_df = pd.DataFrame([student.data])
        
        X_processed = preprocessor.transform(input_df)
        
        prediction_num = model.predict(X_processed)
        
        result_text = label_encoder.inverse_transform(prediction_num)
        
        return {
            "status": "success",
            "prediction": result_text[0]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
