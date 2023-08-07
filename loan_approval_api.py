import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title='Loan Approval System', description='random forest model is used for this project')

# Load the model from the file
loaded_model = joblib.load('best_random_forest_model.pkl')

class Input(BaseModel):
    age: int
    income: float
    employment_status: str
    loan_amount: float
    years_to_return: float

def predict_loan_approval(df: Input):
    # Convert employment_status to numerical representation (assumed mapping)
    employment_status_mapping = {'Employed': 0, 'Unemployed': 1, 'Self-Employed': 2}
    employment_status_numeric = employment_status_mapping.get(df.employment_status, 0)

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'age': [df.age],
        'income': [df.income],
        'employment_status': [employment_status_numeric],
        'loan_amount': [df.loan_amount],
        'years_to_return': [df.years_to_return]
    })

    # Predict loan approval using the model
    prediction = loaded_model.predict(user_data)
    return prediction[0]

@app.post("/predict/")
def predict_loan(df: Input):
        prediction = predict_loan_approval(df)
        if prediction == 1:
            return {"message": "Congratulations! Your loan is approved."}
        else:
            return {"message": "Sorry, your loan is not approved."}
        
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    #uvicorn app:app --reload

    
