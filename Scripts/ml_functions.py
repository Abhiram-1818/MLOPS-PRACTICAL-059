import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error

# Define paths
ARTIFACTS_PATH = "C:/Users/Admin/Desktop/MLOPS  AI22059/theory/mlops2025-DSC/Artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")

def training_pipeline(X_train, y_train):
    """
    Trains an XGBoost classifier and saves the model.
    """
    try:
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        # Save the trained model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        log_info(f"Model trained and saved at {MODEL_PATH}")
        return model
    except Exception as e:
        log_error(f"Error during model training: {e}")
        raise

def load_model():
    """
    Loads the trained model from file.
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        log_info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        log_error(f"Model file not found at {MODEL_PATH}")
        raise

def prediction_pipeline(X_val):
    """
    Makes predictions using the trained model and the label encoder.
    """
    try:
        # Load model and label encoder
        model = pickle.load(open(MODEL_PATH, 'rb'))
        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)

        # Make predictions
        predictions = model.predict(X_val)

        # Ensure predictions are reversed to original labels using label encoder
        predictions = label_encoder.inverse_transform(predictions)

        return predictions
    except Exception as e:
        log_error(f"Error during prediction: {e}")
        raise

def evaluation_matrices(X_val, y_val):
    """
    Evaluates the model using confusion matrix, accuracy, and classification report.
    """
    try:
        # Ensure label encoder and model exist
        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)

        # Get predictions and decode
        pred_vals = prediction_pipeline(X_val)
        decoded_y_vals = label_encoder.inverse_transform(y_val)
        
        # Generate evaluation metrics
        conf_matrix = confusion_matrix(decoded_y_vals, pred_vals, labels=label_encoder.classes_)
        acc_score = accuracy_score(decoded_y_vals, pred_vals)
        class_report = classification_report(decoded_y_vals, pred_vals)
        
        log_info("Model evaluation completed.")
        log_info(f"Confusion Matrix:\n{conf_matrix}")
        log_info(f"Accuracy Score: {acc_score}")
        log_info(f"Classification Report:\n{class_report}")
        
        return conf_matrix, acc_score, class_report
    except FileNotFoundError:
        log_error("Label encoder file not found.")
        raise
    except Exception as e:
        log_error(f"Error during model evaluation: {e}")
        raise
