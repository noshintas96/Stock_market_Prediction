# Stock Market Prediction with MLP and LSTM

In this project, we aim to predict stock market prices using Multilayer Perceptron (MLP) and Long Short-Term Memory (LSTM) neural networks. We will use historical stock data from Amazon (AMZN) to train and evaluate our models.

## Data Collection:

We collect historical stock data for Amazon from reliable sources such as Yahoo Finance or Alpha Vantage.

## Model Implementation:

### Multilayer Perceptron (MLP):
We implement a feedforward neural network with multiple hidden layers.
The input features are fed into the network, followed by hidden layers with activation functions such as ReLU.
The output layer predicts the future stock price.

### Long Short-Term Memory (LSTM):
We implement a type of recurrent neural network (RNN) that is well-suited for sequence prediction tasks.
LSTM networks can capture long-term dependencies in sequential data.
We train the LSTM model using historical stock prices as sequences of data.

## Model Training and Evaluation:

We split the dataset into training and testing sets, reserving a portion of the data for model evaluation.
Both the MLP and LSTM models are trained on the training data and evaluated on the testing data.
Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to assess the model performance.

## Conclusion:

By implementing MLP and LSTM models, we aim to predict future stock prices for Amazon.
The chosen model provides insights into potential stock price trends, aiding investors in decision-making.
