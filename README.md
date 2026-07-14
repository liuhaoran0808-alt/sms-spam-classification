# SMS Spam Classification & Parallel Text Processing

This is an academic Python project that classifies SMS messages as spam or ham using text preprocessing, TF-IDF vectorization, and a Naive Bayes classifier.

The project also compares three text-cleaning approaches:

- Sequential processing
- Multithreading
- Multiprocessing

## Features

- Downloads the SMS Spam Collection dataset from the UCI Machine Learning Repository
- Cleans SMS text by lowercasing, removing URLs, punctuation, and extra whitespace
- Compares sequential, multithreaded, and multiprocessing execution time
- Trains a spam classifier using TF-IDF and Multinomial Naive Bayes
- Prints classification metrics such as precision, recall, F1-score, and accuracy
- Generates a simple chart comparing execution time and model accuracy

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- concurrent.futures

## Project Structure

```text
sms-spam-classification/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── sms_spam_classifier.py
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python src/sms_spam_classifier.py
```

The dataset will be downloaded automatically from the UCI Machine Learning Repository.

## What I Learned

- Basic natural language processing workflow
- Text cleaning and feature extraction using TF-IDF
- Training a basic machine learning classification model
- Comparing sequential, multithreaded, and multiprocessing approaches
- Presenting model performance using classification reports and charts

## Notes

This project was originally developed as a Google Colab academic assignment and later cleaned up for GitHub portfolio use.
