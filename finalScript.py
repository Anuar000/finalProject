import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import wandb

matplotlib.use('TkAgg')

# Initialize WandB
wandb.init(project="regression-classification-task", config={
    "learning_rate": 0.01,
    "epochs": 50,
    "hidden_size": 16,
    "batch_size": 32
})

# Load configuration from WandB
config = wandb.config

# Generate Sample Data
X, y = make_classification(
    n_samples=1000, n_features=2, n_classes=2, n_informative=2,
    n_redundant=0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the generated data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Generated Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Define the Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Initialize the model
input_size = 2
hidden_size = config.hidden_size
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)

# Define the Loss Function and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Prepare Data for PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Train the Model
epochs = config.epochs
batch_size = config.batch_size
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # Mini-batch Gradient Descent
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Log the loss to WandB
    wandb.log({"Epoch": epoch, "Loss": epoch_loss / len(X_train_tensor)})

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Evaluate the Model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).round()
    accuracy = accuracy_score(y_test, predictions.numpy())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    wandb.log({"Test Accuracy": accuracy})

# Visualize the Model's Predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions.numpy().flatten(), cmap='coolwarm', edgecolor='k')
plt.title("Model Predictions on Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Define User Input Prediction Function
def predict_user_data(model, user_input):
    model.eval()
    user_tensor = torch.FloatTensor([user_input])
    prediction = model(user_tensor).round().item()
    return "Class 1" if prediction == 1 else "Class 0"

# Example of User Input Prediction
print("---------------- \n")
print("Write your own numbers: ")
user_input = list(map(float, input().split()))
result = predict_user_data(model, user_input)
print(f"Predicted class for input {user_input}: {result}")
