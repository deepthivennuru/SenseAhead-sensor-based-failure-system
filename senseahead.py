import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)
import pickle

# Preprocess the data
def preprocess_data(data):
    data.columns = data.columns.str.strip().str.lower()
    if 'fault status' not in data.columns:
        raise KeyError("The 'Fault Status' column is missing from the dataset.")
    X = data.drop(columns=['fault status'], errors='ignore')
    y = data['fault status']
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
    X = X.fillna(0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Encode string labels to binary values
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)  # Converts labels to 0 and 1
    
    # Predict and encode predictions to match y_test_encoded
    y_pred = model.predict(X_test)
    y_pred_encoded = label_encoder.transform(y_pred)  # Ensure predictions are encoded the same way
    
    # Predict probabilities for ROC curve (if applicable)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
        'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
        'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_pred_encoded),
        'roc_curve': roc_curve(y_test_encoded, y_prob) if y_prob is not None else None,
        'auc': auc(*roc_curve(y_test_encoded, y_prob)[:2]) if y_prob is not None else None
    }
    return metrics
# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Plot ROC curve
def plot_roc_curve(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(fig)

# Main function
def main():
    st.title("SenseAhead: Machine Maintenance Predictor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your sensor data CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        
        # Display dataset preview
        st.write("### Dataset Preview:")
        st.write(data.head())
        
        # Display class distribution
        st.write("### Class Distribution:")
        st.bar_chart(data['Fault Status'].value_counts())
        
        # Preprocess the data
        try:
            X_train, X_test, y_train, y_test = preprocess_data(data)
        except KeyError as e:
            st.error(str(e))
            return
        
        # Train the model
        try:
            model = train_model(X_train, y_train)
        except ValueError as e:
            st.error(f"Error during model training: {e}")
            return
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        st.success(f"Model trained successfully!")
        st.write(f"### Performance Metrics:")
        st.write(f"- Accuracy: {metrics['accuracy']:.2f}")
        st.write(f"- Precision: {metrics['precision']:.2f}")
        st.write(f"- Recall: {metrics['recall']:.2f}")
        st.write(f"- F1 Score: {metrics['f1_score']:.2f}")
        
        # Plot confusion matrix
        st.write("### Confusion Matrix:")
        class_names = np.unique(y_test)
        plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        
        # Plot ROC curve (if applicable)
        if metrics['roc_curve'] is not None:
            st.write("### ROC Curve:")
            fpr, tpr, _ = metrics['roc_curve']
            plot_roc_curve(fpr, tpr, metrics['auc'])
        
        # Feature importance
        st.write("### Feature Importance:")
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(feature_importances)
        st.bar_chart(feature_importances.set_index('Feature'))
        
        # Download the model
        st.write("### Download Trained Model:")
        model_file = "trained_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        with open(model_file, "rb") as f:
            st.download_button("Download Model", f, file_name="trained_model.pkl")

if __name__ == "__main__":
    main()