import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Naive Bayes Classifier Function
# ==========================================
def naive_bayes_classifier(X, y, test_size=0.2, random_state=42):
    """
    Trains a Naive Bayes classifier and returns comprehensive metrics.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (dataset features)
    y : array-like
        Target variable (labels)
    test_size : float
        Proportion of dataset to include in test split (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing model and performance metrics
    """
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create a Gaussian Naive Bayes classifier
    model = GaussianNB()
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Predict labels for training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate training accuracy
    training_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Calculate testing accuracy
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Calculate confusion matrix for test set
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Return comprehensive results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'training_accuracy': training_accuracy,
        'testing_accuracy': testing_accuracy,
        'confusion_matrix': conf_matrix,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return results

st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")
st.title("🤖 Naive Bayes Classifier")

# Sidebar for configuration
st.sidebar.header("Configuration")

# ==========================================
# 1. CSV File Upload
# ==========================================
st.sidebar.subheader("1. Dataset Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = st.sidebar.checkbox("Show dataset preview", value=True)
    data = pd.read_csv(uploaded_file)
    
    if df:
        st.subheader("📊 Dataset Preview")
        st.write(f"Dataset shape: {data.shape}")
        st.dataframe(data.head(10))
    
    # ==========================================
    # 2. Feature and Target Selection
    # ==========================================
    st.sidebar.subheader("2. Column Selection")
    columns = data.columns.tolist()
    
    target_column = st.sidebar.selectbox(
        "Select target column (y)",
        columns,
        help="The column containing labels/target values"
    )
    
    feature_columns = st.sidebar.multiselect(
        "Select feature columns (X)",
        [col for col in columns if col != target_column],
        default=[col for col in columns if col != target_column],
        help="The columns containing features/predictors"
    )
    
    if not feature_columns:
        st.error("Please select at least one feature column!")
        st.stop()
    
    # ==========================================
    # 3. Training/Testing Split
    # ==========================================
    st.sidebar.subheader("3. Train-Test Split")
    test_size = st.sidebar.slider(
        "Test set size (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Percentage of data to use for testing"
    )
    test_size = test_size / 100
    
    # ==========================================
    # 4. Random State
    # ==========================================
    random_state = st.sidebar.number_input(
        "Random seed",
        value=42,
        help="For reproducibility"
    )
    
    # ==========================================
    # Data Preprocessing
    # ==========================================
    st.sidebar.subheader("4. Data Preprocessing")
    
    # Extract features and target
    X = data[feature_columns].copy()
    y = data[target_column].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected in features!")
        handle_missing = st.sidebar.radio(
            "How to handle missing values?",
            ["Drop rows", "Fill with mean"]
        )
        if handle_missing == "Drop rows":
            X = X.dropna()
            y = y[X.index]
        else:
            X = X.fillna(X.mean())
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(categorical_cols) > 0:
        st.info(f"Encoding categorical columns: {', '.join(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
    # Convert to numpy arrays
    X = X.values
    y = np.array(y)
    
    # ==========================================
    # Train Model
    # ==========================================
    st.sidebar.subheader("5. Model Training")
    if st.sidebar.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training Naive Bayes classifier..."):
            # Train the model
            results = naive_bayes_classifier(
                X=X,
                y=y,
                test_size=test_size,
                random_state=int(random_state)
            )
        
        st.success("✅ Model training completed!")
        
        # Display Results
        st.subheader("📈 Model Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Training Accuracy",
                f"{results['training_accuracy']:.4f}",
                f"{results['training_accuracy']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Testing Accuracy",
                f"{results['testing_accuracy']:.4f}",
                f"{results['testing_accuracy']*100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Train-Test Samples",
                f"{results['train_size']}/{results['test_size']}"
            )
        
        # Display Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        
        conf_matrix = results['confusion_matrix']
        
        # Validate confusion matrix
        if conf_matrix is None or len(conf_matrix) == 0:
            st.error("Confusion matrix is empty!")
        else:
            # Create a more detailed view
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Prepare labels based on matrix shape
                heatmap_kwargs = {
                    'data': conf_matrix,
                    'annot': True,
                    'fmt': 'd',
                    'cmap': 'Blues',
                    'cbar': True,
                    'ax': ax
                }
                
                if conf_matrix.shape[0] == 2:
                    heatmap_kwargs['xticklabels'] = ['Negative', 'Positive']
                    heatmap_kwargs['yticklabels'] = ['Negative', 'Positive']
                
                sns.heatmap(**heatmap_kwargs)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix Heatmap')
                st.pyplot(fig)
            
            with col2:
                # Text representation
                st.text("Confusion Matrix:")
                st.code(str(conf_matrix))
                
                # Calculate additional metrics
                if conf_matrix.shape[0] == 2:  # Binary classification
                    tn, fp, fn, tp = conf_matrix.ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    st.subheader("📊 Additional Metrics (Binary Classification)")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Precision", f"{precision:.4f}")
                    with metric_col2:
                        st.metric("Recall", f"{recall:.4f}")
                    with metric_col3:
                        st.metric("F1-Score", f"{f1:.4f}")
        
        # Display configuration used
        st.subheader("⚙️ Configuration Used")
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.write("**Features Used:**")
            st.write(feature_columns)
        
        with config_col2:
            st.write("**Target Column:**")
            st.write(target_column)
        
        with config_col3:
            st.write("**Split Configuration:**")
            st.write(f"Train: {int((1-test_size)*100)}%")
            st.write(f"Test: {int(test_size*100)}%")

else:
    st.info("👈 Upload a CSV file from the sidebar to get started!")
    
    # Show example of expected CSV format
    with st.expander("ℹ️ Expected CSV Format"):
        st.write("Your CSV file should have columns for features and a target column.")
        example_data = pd.DataFrame({
            'Feature1': [1.5, 2.3, 3.1],
            'Feature2': [4.2, 5.1, 6.3],
            'Feature3': [7.0, 8.5, 9.2],
            'Target': ['Class_A', 'Class_B', 'Class_A']
        })
        st.dataframe(example_data)
