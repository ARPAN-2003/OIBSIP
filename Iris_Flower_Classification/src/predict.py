import joblib
import pandas as pd

def predict_flower(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width):

    model = joblib.load("models/iris_model.pkl")

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "Sepal Length (Cm)",
            "Sepal Width (Cm)",
            "Petal Length (Cm)",
            "Petal Width (Cm)"
        ]
    )

    prediction = model.predict(sample)

    return prediction[0]