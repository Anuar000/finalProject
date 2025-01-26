# Final project
Made by Anuar Dosmaganbetov and Aliza Maulenkul

# Binary Classification with PyTorch

A simple project to train and evaluate a neural network for binary classification using PyTorch. The data is generated automatically, and results are tracked with [Weights & Biases (W&B)](https://wandb.ai/).

## Features
- Generate synthetic data using `sklearn`.
- Visualize data and predictions with `matplotlib`.
- Build and train a neural network in PyTorch.
- Log metrics with W&B.
- Make predictions for custom user input.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/binary-classification-pytorch.git
   cd binary-classification-pytorch
   ```
2. Start the project:
   ```bash
   python main.py
   ```
3. (Optional) Log in to W&B to track metrics:
   ```bash
   wandb login
   ```

## Usage
After training the model, you can input your own data to get predictions:
```bash
Enter two feature values separated by space: 1.5 -2.3
Predicted class for [1.5, -2.3]: Class 1
```

## Visualization
- **Data**: Shown as points from two classes on a scatter plot.
- **Results**: After training, the model displays predictions for the test data.


## Requirements
- Python 3.8+
- PyTorch 1.10+
- scikit-learn
- matplotlib
- Weights & Biases (`wandb`)

You can install all the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Training Process
This project uses a simple neural network architecture with one hidden layer. The training process involves:
1. **Generating synthetic data** for binary classification using `sklearn.datasets.make_classification`.
2. **Splitting the dataset** into training and testing sets (80%/20%).
3. **Training the neural network** with mini-batch gradient descent and the Adam optimizer.
4. **Tracking the training progress** (loss and accuracy) using Weights & Biases (W&B).
5. **Evaluating the model** on the test set and displaying the accuracy.

## Visualizations
During training, the following visualizations are available:
- **Loss curve**: Training loss per epoch.
- **Accuracy**: Test accuracy logged on W&B.
- **Scatter plot** of data and predictions:
    - Shows the distribution of synthetic data.
    - Displays model predictions on the test data, visualized with color-coded points.

## Example Output
After training the model, you can input custom data to predict the class:
```bash
Enter two feature values separated by space: 1.5 -2.3
Predicted class for [1.5, -2.3]: Class 1
```

