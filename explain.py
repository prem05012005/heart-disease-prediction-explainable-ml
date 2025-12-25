import numpy as np
import pandas as pd

def explain_prediction(model, scaler, X, feature_names, sample_index=0):
    """
    Explain feature contributions for one prediction
    """

    # Original sample
    x_original = X.iloc[sample_index].values.reshape(1, -1)
    x_original_scaled = scaler.transform(x_original)

    # Base prediction probability
    base_pred = model.predict_proba(x_original_scaled)[0][1]

    contributions = {}

    for i, feature in enumerate(feature_names):
        x_modified = x_original.copy()

        # Replace feature with its mean value
        x_modified[0][i] = X.iloc[:, i].mean()

        x_modified_scaled = scaler.transform(x_modified)
        new_pred = model.predict_proba(x_modified_scaled)[0][1]

        contributions[feature] = base_pred - new_pred

    return base_pred, contributions
