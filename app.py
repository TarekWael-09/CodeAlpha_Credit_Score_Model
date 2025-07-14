import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Scoring Prediction",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Scoring Prediction System")
st.markdown("### Quick Credit Score Prediction")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

def create_sample_data():
    """Create sample data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Occupation': np.random.choice(['Engineer', 'Scientist', 'Teacher', 'Doctor', 'Manager'], n_samples),
        'Annual_Income': np.random.randint(20000, 200000, n_samples),
        'Monthly_Inhand_Salary': np.random.randint(2000, 15000, n_samples),
        'Num_Bank_Accounts': np.random.randint(1, 10, n_samples),
        'Num_Credit_Card': np.random.randint(0, 8, n_samples),
        'Interest_Rate': np.random.randint(4, 35, n_samples),
        'Num_of_Loan': np.random.randint(0, 10, n_samples),
        'Credit_Mix': np.random.choice(['Good', 'Standard', 'Poor'], n_samples),
        'Outstanding_Debt': np.random.randint(0, 50000, n_samples),
        'Credit_Utilization_Ratio': np.random.randint(10, 50, n_samples),
        'Payment_of_Min_Amount': np.random.choice(['Yes', 'No'], n_samples),
        'Total_EMI_per_month': np.random.randint(0, 5000, n_samples),
        'Amount_invested_monthly': np.random.randint(0, 10000, n_samples),
        'Payment_Behaviour': np.random.choice(['Low_spent_Small_value_payments', 'High_spent_Small_value_payments', 
                                            'Low_spent_Medium_value_payments', 'High_spent_Medium_value_payments',
                                            'Low_spent_Large_value_payments', 'High_spent_Large_value_payments'], n_samples),
        'Monthly_Balance': np.random.randint(-1000, 10000, n_samples),
        'Delay_from_due_date': np.random.randint(0, 30, n_samples),
        'Predicted_Credit_Score': np.random.choice(['Good', 'Standard', 'Poor'], n_samples)
    }
    
    return pd.DataFrame(data)

def preprocess_data(df, target_column=None):
    """Preprocess the data for training"""
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Auto-detect target column if not provided
    if target_column is None:
        possible_target_cols = ['Predicted_Credit_Score', 'Predicted_Credit_Score', 'credit_score', 'target', 'label']
        for col in possible_target_cols:
            if col in df.columns:
                target_column = col
                break
    
    # Create label encoders for categorical variables
    encoders = {}
    categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    
    for col in categorical_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
    
    # Create synthetic credit scores if no target column
    if target_column is None:
        def create_synthetic_score(row):
            score = 0
            # Income factor
            if row.get('Annual_Income', 0) > 50000:
                score += 2
            elif row.get('Annual_Income', 0) > 30000:
                score += 1
            
            # Age factor
            if row.get('Age', 0) > 30:
                score += 1
            
            # Bank accounts factor
            if row.get('Num_Bank_Accounts', 0) >= 2:
                score += 1
            
            # Credit cards factor
            if row.get('Num_Credit_Card', 0) <= 3:
                score += 1
            
            # Interest rate factor
            if row.get('Interest_Rate', 0) < 20:
                score += 1
            
            # Delay factor
            if row.get('Delay_from_due_date', 0) < 5:
                score += 1
            
            if score >= 5:
                return 'Good'
            elif score >= 3:
                return 'Standard'
            else:
                return 'Poor'
        
        df['Synthetic_Credit_Score'] = df.apply(create_synthetic_score, axis=1)
        target_column = 'Synthetic_Credit_Score'
    
    # Encode target variable
    if target_column in df.columns:
        encoders[target_column] = LabelEncoder()
        df[target_column] = encoders[target_column].fit_transform(df[target_column].astype(str))
    
    return df, encoders, target_column

def train_model(df, target_column=None):
    """Train the credit scoring model"""
    df_processed, encoders, detected_target = preprocess_data(df, target_column)
    
    # Separate features and target
    X = df_processed.drop(detected_target, axis=1)
    y = df_processed[detected_target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, encoders, scaler, X.columns, accuracy, detected_target

# Auto-load and train model on app start
if st.session_state.model is None:
    with st.spinner("Loading model... Please wait"):
        try:
            # Load sample data and train model
            df = create_sample_data()
            model, encoders, scaler, feature_names, accuracy, target_col = train_model(df)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.encoders = encoders
            st.session_state.scaler = scaler
            st.session_state.feature_names = feature_names
            st.session_state.target_column = target_col
            
            st.success(f"✅ Model ready! Accuracy: {accuracy:.2f}")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

# File upload section (optional)
with st.expander("📁 Upload Your Own Data (Optional)"):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data uploaded successfully!")
        
        if st.button("🔄 Retrain Model with Your Data"):
            with st.spinner("Retraining model with your data..."):
                try:
                    model, encoders, scaler, feature_names, accuracy, target_col = train_model(df)
                    
                    # Update session state
                    st.session_state.model = model
                    st.session_state.encoders = encoders
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = feature_names
                    st.session_state.target_column = target_col
                    
                    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error retraining model: {str(e)}")

# Main prediction section
st.header("🎯 Predict Credit Score")

if st.session_state.model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", ["Engineer", "Scientist", "Teacher", "Doctor", "Manager"])
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        monthly_salary = st.number_input("Monthly Salary ($)", min_value=0, value=4000, step=100)
        
        st.subheader("Banking Information")
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=2)
        num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, value=3)
        monthly_balance = st.number_input("Monthly Balance ($)", value=5000, step=100)
        
        st.subheader("Credit Information")
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, value=15)
        num_loans = st.number_input("Number of Loans", min_value=0, value=1)
        credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Poor"])
        
    with col2:
        st.subheader("Financial Behavior")
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, value=10000, step=500)
        credit_utilization = st.number_input("Credit Utilization Ratio (%)", min_value=0, value=25)
        payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No"])
        
        st.subheader("Monthly Payments")
        total_emi = st.number_input("Total EMI per month ($)", min_value=0, value=1000, step=50)
        amount_invested = st.number_input("Amount Invested Monthly ($)", min_value=0, value=2000, step=100)
        
        st.subheader("Payment Pattern")
        payment_behaviour = st.selectbox("Payment Behaviour", 
                                       ["Low_spent_Small_value_payments", "High_spent_Small_value_payments",
                                        "Low_spent_Medium_value_payments", "High_spent_Medium_value_payments",
                                        "Low_spent_Large_value_payments", "High_spent_Large_value_payments"])
        
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, value=3)
    
    # Predict button
    if st.button("🎯 Predict Credit Score", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Age': age,
                'Occupation': occupation,
                'Annual_Income': annual_income,
                'Monthly_Inhand_Salary': monthly_salary,
                'Num_Bank_Accounts': num_bank_accounts,
                'Num_Credit_Card': num_credit_cards,
                'Interest_Rate': interest_rate,
                'Num_of_Loan': num_loans,
                'Credit_Mix': credit_mix,
                'Outstanding_Debt': outstanding_debt,
                'Credit_Utilization_Ratio': credit_utilization,
                'Payment_of_Min_Amount': payment_min_amount,
                'Total_EMI_per_month': total_emi,
                'Amount_invested_monthly': amount_invested,
                'Payment_Behaviour': payment_behaviour,
                'Monthly_Balance': monthly_balance,
                'Delay_from_due_date': delay_from_due_date
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            encoders = st.session_state.encoders
            for col in ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col])
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            target_col = st.session_state.target_column
            credit_score = encoders[target_col].inverse_transform([prediction])[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if credit_score == 'Good':
                    st.success(f"**Predicted_Credit_Score: {credit_score}** ✅")
                elif credit_score == 'Standard':
                    st.warning(f"**Predicted_Credit_Score: {credit_score}** ⚠️")
                else:
                    st.error(f"**Predicted_Credit_Score: {credit_score}** ❌")
            
            with col2:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col3:
                risk_level = "Low" if credit_score == 'Good' else "Medium" if credit_score == 'Standard' else "High"
                st.metric("Risk Level", risk_level)
            
            # Probability chart
            col1, col2 = st.columns(2)
            
            with col1:
                classes = encoders[target_col].classes_
                prob_df = pd.DataFrame({
                    'Credit Score': classes,
                    'Probability': prediction_proba
                })
                
                fig = px.bar(prob_df, x='Credit Score', y='Probability', 
                           title="Prediction Probabilities",
                           color='Credit Score',
                           color_discrete_map={'Good': 'green', 'Standard': 'orange', 'Poor': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Recommendations
                st.subheader("💡 Recommendations")
                if credit_score == 'Good':
                    st.write("✅ Excellent credit profile!")
                    st.write("• Maintain current payment habits")
                    st.write("• Consider increasing investments")
                elif credit_score == 'Standard':
                    st.write("⚠️ Room for improvement:")
                    st.write("• Reduce credit utilization")
                    st.write("• Pay bills on time")
                    st.write("• Consider debt consolidation")
                else:
                    st.write("❌ Action needed:")
                    st.write("• Reduce outstanding debt")
                    st.write("• Improve payment consistency")
                    st.write("• Seek financial counseling")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your input values and try again.")

else:
    st.warning("⚠️ Model not loaded. Please refresh the page.")

# Footer
st.markdown("---")
st.markdown("💳 **Credit Scoring Prediction System** - Built with Streamlit")