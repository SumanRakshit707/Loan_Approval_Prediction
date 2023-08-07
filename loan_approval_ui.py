import streamlit as st
import pandas as pd
import joblib

# Load the model from the file
# Save the model to a file
model_filename = 'best_random_forest_model.pkl'
loaded_model = joblib.load(model_filename)


def predict_loan_approval(df):
    # Make predictions on the DataFrame
    predictions = loaded_model.predict(df)
    return predictions

def main():
    st.title('Loan Approval Prediction')
    st.write("Enter your details to get the loan approval prediction:")

    # Create input fields for user data
    age = st.text_input('Age')
    income = st.text_input('Income')
    employment_status = st.selectbox('Employment Status', ['Employed', 'Unemployed', 'Self-Employed'])
    loan_amount = st.text_input('Loan Amount')
    years_to_return = st.text_input('Years to Return the Loan')

    # Convert employment_status to numerical representation (assumed mapping)
    employment_status_mapping = {'Employed': 0, 'Unemployed': 1, 'Self-Employed': 2}
    employment_status_numeric = employment_status_mapping.get(employment_status, 0)

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'employment_status': [employment_status_numeric],
        'loan_amount': [loan_amount],
        'years_to_return': [years_to_return]
    })

    # Predict loan approval using the model
    if st.button('Predict Loan Approval'):
        prediction = predict_loan_approval(user_data)
        if prediction[0] == 1:
            st.success('Congratulations! Your loan is approved.')
        else:
            st.error('Sorry, your loan is not approved.')

if __name__ == '__main__':
    main()
