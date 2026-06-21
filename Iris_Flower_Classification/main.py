from src.train import train_model
from src.predict import predict_flower
from src.visualization import generate_visualizations

# Generate charts
generate_visualizations("data/Iris.csv")

# Train model
train_model("data/Iris.csv")

# Predict flower
result = predict_flower(
    5.1,
    3.5,
    1.4,
    0.2
)

print("\nPredicted Flower Species:")
print(result)