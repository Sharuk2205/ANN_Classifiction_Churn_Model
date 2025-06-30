
# 🧠 Customer Churn Prediction with Artificial Neural Networks

This project involves building and deploying an **Artificial Neural Network (ANN)** to predict whether a customer is likely to leave a bank. It consists of two parts:

- A Jupyter Notebook for training and evaluating the ANN model.
- A Streamlit web application for real-time churn prediction using the trained model.

  ![image](https://github.com/user-attachments/assets/3f6301f1-d32c-4a46-81d7-40eae9ec2c16)




---

## 📁 Files

| File Name                                               | Description                                                |
|---------------------------------------------------------|------------------------------------------------------------|
| `artificial_neural_network(Customer Churn Prediction).ipynb` | Notebook with preprocessing, model training, and evaluation |
| `Churn_Modelling.csv`                                   | Dataset containing customer details                         |
| `model.h5`                                               | Trained ANN model saved in Keras format                    |
| `scaler.pkl`                                             | StandardScaler used for feature scaling                    |
| `label_encoder_gender.pkl`                              | LabelEncoder for encoding the `Gender` feature             |
| `onehot_encoder_geo.pkl`                                | OneHotEncoder for encoding the `Geography` feature         |
| `app.py`                                                | Streamlit app for interactive predictions                  |

---

## ⚙️ Workflow

### 1. 📦 Importing Libraries
Essential libraries like `NumPy`, `Pandas`, `TensorFlow`, `Keras`, and `scikit-learn` are imported.

### 2. 🧹 Data Preprocessing
- Load `Churn_Modelling.csv`
- Drop irrelevant columns (e.g., `RowNumber`, `CustomerId`, `Surname`)
- Encode categorical features:
  - `Gender` with `LabelEncoder`
  - `Geography` with `OneHotEncoder` (excluding one dummy variable to avoid dummy trap)
- Split into training and test sets
- Feature scaling using `StandardScaler`

### 3. 🏗️ Building the ANN
- Use `Sequential` API from Keras
- Architecture:
  - Input layer
  - Two hidden layers with ReLU activation
  - Output layer with sigmoid activation (binary classification)

### 4. 🏋️ Training the ANN
- Optimizer: `Adam`
- Loss: `binary_crossentropy`
- Epochs: 100
- Batch size: 32

### 5. 📈 Model Evaluation
- Evaluate on test set using:
  - Confusion Matrix
  - Accuracy Score
- Achieved **~86.3% accuracy**

### 6. 🔮 Making Predictions
- Use the trained model to predict a single new observation
- Ensure input format is `[[...]]`
- Input features must match training order and preprocessing

---

## 🧬 Streamlit App Pipeline

1. **User Input**: Collected via UI form (geography, age, salary, etc.)
2. **Encoding**:
   - `Gender` is label-encoded
   - `Geography` is one-hot encoded
3. **Data Merge**: All features combined into a single DataFrame
4. **Scaling**: Scaled using the pre-fitted `StandardScaler`
5. **Prediction**: Passed to `model.h5` for churn prediction
6. **Output**: Returns churn probability and final decision

---

## 💻 Run the App

### Option 1: Google Colab
- Open the `.ipynb` file in [Google Colab](https://colab.research.google.com)
- Run all cells in order

### Option 2: Local Environment

#### 1. Clone the Repo

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

#### 2. Create a Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate        # Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

#### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Example Prediction

Predicting for the following customer:

- Geography: France
- Gender: Female
- Age: 42
- Tenure: 3
- Balance: 60000
- NumOfProducts: 2
- HasCrCard: 1
- IsActiveMember: 1
- EstimatedSalary: 50000

Model Output:
```
Churn Probability: 0.27
The customer is not likely to churn.
```

---

## 📝 Notes

- Ensure encoders and scaler are used **in the same order** as in training.
- All `.pkl` and `.h5` files must be present in the working directory for the app to work.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Questions or Suggestions?

Feel free to open an issue or submit a pull request!
