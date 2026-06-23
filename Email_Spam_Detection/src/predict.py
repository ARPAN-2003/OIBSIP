import joblib
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from preprocess import transform_text

model=joblib.load(
    "models/spam_model.pkl"
)

vectorizer=joblib.load(
    "models/vectorizer.pkl"
)

def predict_sms(message):

    message=transform_text(message)

    vector=vectorizer.transform([message])

    prediction=model.predict(vector)[0]

    probability=model.predict_proba(vector)[0]

    spam_probability=float(probability[1])

    ham_probability=float(probability[0])

    if prediction==1:
        label="SPAM"

    else:
        label="HAM"

    return (
        label,
        spam_probability,
        ham_probability
    )