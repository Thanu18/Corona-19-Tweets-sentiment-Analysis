# Sentiment Analysis on Tweets

## Project Description

This project involves sentiment analysis of tweets related to COVID-19. The dataset contains tweets with sentiments classified as Negative, Neutral, or Positive. Various preprocessing techniques and machine learning models, including Naive Bayes, Logistic Regression, K-Nearest Neighbors (KNN), LSTM, BERT, and RoBERTa, are employed to classify these sentiments. The goal is to develop and evaluate these models for sentiment classification.

## Project Structure

- **Data Preparation**: Cleaning and preprocessing of tweet data.
- **Feature Extraction**: Using methods like TF-IDF and tokenization.
- **Model Training and Evaluation**: Implementing and evaluating Naive Bayes, Logistic Regression, KNN, LSTM, BERT, and RoBERTa models.
- **Results**: Performance metrics, classification reports, and confusion matrices.
- **Comparing the models**: Compare Traditional models, Deeplearing (LSTM and Transformer Models)

## Installation

Clone the repository and install the required packages using pip:

```bash
git clone https://github.com/your-username/sentiment-analysis-tweets.git
cd sentiment-analysis-tweets
pip install -r requirements.txt

```

## Data Preprocessing

1. Data Collection:
- This Covid-19 tweets dataset is collected from Kaggle
- In the format of CSV file
- Combined the train and test data into one dataframe

2. Text Cleaning:
- Lowercasing: Convert all text to lowercase to maintain consistency.
- Removing Punctuation and Special Characters: Strip out punctuation, special characters, and numbers that are not relevant to the analysis.
- Removing URLs, User Mentions, and Hashtags: For tweet analysis, URLs, mentions (@username), and hashtags (#hashtag) might be removed or handled separately depending on the goal.

3. Tokeization:
- Breaking Down Text: Split text into individual words or tokens. This step helps in creating a bag-of-words or other representations of the text.
- Libraries: Nltk library is used for tokenization

4. Stop Words Removal:
- Filtering Out Common Words: Remove common words (e.g., "and," "the") that may not contribute meaningful information for sentiment analysis.
- Libraries: Use NLTK’s list of stop words or create a custom list.

5. Lemmatization/Stemming:
- Reducing Words to Base Form: Lemmatization and stemming reduce words to their base or root form. For example, "running" becomes "run."
- Libraries: Used NLTK for lemmatization and stemming.

6. Handling Imbalanced Data:
- Balancing Classes: The dataset has imbalanced classes (e.g., more positive tweets than negative), Used Random Over sampling

7. Padding Sequences:
For Neural Networks: When working with sequences of text (e.g., sentences), ensure that all sequences are of the same length by padding shorter sequences with zeros or truncating longer ones.

## Building Model

Define the LSTM Model:

Function create_lstm_model(vocab_size, max_length):
Embedding layer: Transforms word indices into dense vectors.
LSTM layer: Captures temporal dependencies with 128 units.
Dense layer: Produces class probabilities with softmax activation.
Compiled with categorical crossentropy loss and Adam optimizer.

Prepare Text Data:

Tokenization:
Initialize Tokenizer with a vocabulary limit of 5000 words and an OOV token.
Fit on the training texts to create a word index and determine vocabulary size.
Padding:
Determine the maximum sequence length from training data.
Convert texts to sequences and pad them to ensure uniform length for training, validation, and testing datasets.
Train the Model:

Fit the LSTM model on padded training data for 10 epochs.
Validate the model using the validation dataset.
Evaluate the Model:

Assess model performance on the test dataset by computing loss and accuracy.
Make Predictions:

Predict class probabilities on the test data.
Convert probabilities to class labels.
Analyze Results:

Print a classification report showing precision, recall, and F1-score.
Plot a confusion matrix to visualize classification performance.

![Image](Pics\ConfusionMatrix.png)
![Image](Pics\Accuracy.png)

## Make Predictions

### Preprocessing before predicting:

-Strip Entities: Remove URLs, mentions, hashtags, and other special entities.

-Clean Hashtags: Remove or clean up hashtags.

-Strip Emoji: Remove emoji characters.

-Filter Characters: Remove unwanted characters or symbols.

-Remove Multiple Spaces: Replace multiple spaces with a single space.

-Remove Stop Words: Filter out common stop words that do not contribute to meaning.

-Lemmatize: Convert words to their base or root form

-Model Prediction: Use the model to predict the class of the preprocessed and padded text.

-Get Predicted Class: Determine the most likely class from the model's prediction.

### Tokenize and Pad:

- Tokenize: Convert the cleaned and processed text into numerical sequences.

- Pad Sequences: Ensure sequences have a consistent length by padding them.

### Output

<img src="Pics\Prediction.png" alt="Example Image" width="500" height="400">

## Created Falsk Prediction APP

### Setup Flask Environment:
Install Flask: pip install flask.

### Create Flask App (app.py):
Set up the Flask server with routes for the home page and prediction.
Load your pre-trained LSTM model and tokenizer.
Handle text input, preprocess it (tokenize and pad), and make predictions.
Render HTML templates with prediction results.

### Create HTML Templates:
index.html: Provides a form for users to input text.

Negative Prediction

 <img src="Pics\Neg.png" alt="Example Image" width="400" height="300">

Positive Prediction

<img src="Pics\positive.png" alt="Example Image" width="400" height="300">




