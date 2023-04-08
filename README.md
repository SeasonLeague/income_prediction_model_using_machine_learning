# Income Prediction Model
---
This repository contains a neural network model for predicting income levels (above or below $50,000) based on several input features including age, education level, and capital gains and losses.

### Dataset
---
The model was trained on the "income.csv" dataset, which contains 48,842 samples with seven features: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, and income_level. The dataset is available in this repository in the "data" folder.

### Dependencies
---
The following Python libraries are required to run the income prediction model:

- numpy
- pandas
- scikit-learn
- keras

You can install these libraries by running the following command:

~~~
$ pip install numpy pandas scikit-learn keras
~~~


### Usage
---
To use the income prediction model, follow these steps:

1. Clone this repository to your local machine.

2. Open the "income_prediction.ipynb" file in a Jupyter notebook environment.

3. Run the code cells in the notebook to train the neural network model.

4. Once the model is trained, you can use it to make predictions on new data. To do so, load the saved model using the load_model function from Keras and preprocess the input data using the StandardScaler from scikit-learn. Then, use the predict function on the loaded model to make predictions.


~~~
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('income_prediction_model.h5')

# Preprocess the input data
input_data = [[35, 40000, 12, 0, 0, 40]]
sc = StandardScaler()
input_data = sc.fit_transform(input_data)

# Make predictions
prediction = model.predict(input_data)
if prediction > 0.5:
    print("Predicted income level is above threshold.")
else:
    print("Predicted income level is below threshold.")
~~~

### Evaluation
---
The trained income prediction model achieved an accuracy of 83.16% on the test set. Additional evaluation metrics such as precision, recall, and F1 score can be calculated using the provided code in the "income_prediction.ipynb" notebook.