import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.visualization import correlation_heatmap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def train_model(filepath):

    df = pd.read_csv(filepath)
    
    correlation_heatmap(df)

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    categorical_cols = [
        "Car_Name",
        "Fuel_Type",
        "Selling_type",
        "Transmission"
    ]

    numerical_cols = [
        "Year",
        "Present_Price",
        "Driven_kms",
        "Owner"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            ),
            (
                "num",
                "passthrough",
                numerical_cols
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)
    
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        pipeline.named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.title(
        "Top 10 Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()

    predictions = pipeline.predict(X_test)
    
    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        predictions
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(
        "Actual vs Predicted Price"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/prediction_vs_actual.png"
    )

    plt.close()

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    
    with open("outputs/model_metrics.txt", "w") as file:

        file.write(
            f"MAE  : {mae:.4f}\n"
        )

        file.write(
            f"RMSE : {rmse:.4f}\n"
        )

        file.write(
            f"R²   : {r2:.4f}\n"
        )

    joblib.dump(
        pipeline,
        "models/car_price_model.pkl"
    )

    print("\nModel saved successfully.")

    return pipeline