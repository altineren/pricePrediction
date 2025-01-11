# Car Price Prediction using ANN

This project predicts car prices using an Artificial Neural Network (ANN) implemented in Python. The dataset used for training and testing the model is sourced from Kaggle, and the project was developed using Jupyter Notebook with TensorFlow and other data preprocessing libraries.

## Dataset
- **Source**: Kaggle
- **Description**: The dataset contains various features related to cars, including:
  - Manufacturing year
  - Mileage
  - Engine capacity
  - Fuel type
  - Number of owners
  - Selling price (target variable)

## Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values**:
  - Missing values in the dataset were handled by either filling or removing them.
- **Feature Scaling**:
  - Numerical features were normalized using `PowerTransformer` to ensure Gaussian distribution.
- **Feature Selection**:
  - The most important features were selected using Random Forest feature importance analysis.

### 2. ANN Model
- **Architecture**:
  - Input Layer: Equal to the number of selected features.
  - Hidden Layers:
    - Layer 1: 64 neurons, ReLU activation.
    - Layer 2: 32 neurons, ReLU activation.
    - Layer 3: 32 neurons, ReLU activation.
    - Layer 4: 16 neurons, ReLU activation.
  - Output Layer: 1 neuron with linear activation (regression task).
- **Optimization**:
  - Optimizer: Adam optimizer with a learning rate of 0.001.
  - Loss Function: Mean Squared Error (MSE).
  - Metrics: Mean Absolute Error (MAE).

### 3. Training and Evaluation
- **Data Splitting**:
  - 80% of the data was used for training and 20% for testing.
- **Training**:
  - The model was trained for 180 epochs with a batch size of 32.
- **Evaluation**:
  - Evaluated on both training and testing datasets.
  - Metrics include MSE, MAE, and Mean Absolute Percentage Error (MAPE).

## Tools and Libraries
- **Jupyter Notebook**: Used for development and experimentation.
- **TensorFlow/Keras**: For building and training the ANN model.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For preprocessing, feature selection, and splitting the data.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook ProjectAnn.ipynb
   ```

4. Run the notebook cells sequentially to preprocess the data, train the model, and evaluate it.

## Results
- The model predicts car prices with reasonable accuracy.
- Performance metrics like Training Loss, Test Loss, and Accuracy are calculated to evaluate the model.

## Future Improvements
- Experiment with additional hyperparameter tuning and different neural network architectures.
- Add external data sources or create new features to improve prediction accuracy.
- Explore transfer learning or ensemble methods for better results.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

**Note**: The dataset used in this project is publicly available on Kaggle. Please ensure proper attribution if you use it in your work.
