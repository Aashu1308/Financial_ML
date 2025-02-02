# Financial Planning Assistant

### ML Model to help classify financial decisions and improve planning and literacy

#### Overview

This project aims to develop a machine learning model that can classify financial decisions and provide insights to improve financial literacy.

Steps so far:

1. Augmented Data and up and downsampled as necessary
2. Cleaned empty and nonsensical columns
3. Separated into upper_income dataset and lower income dataset
4. Labelled datasets with good and bad spending habits by monthly spending
5. Implemented and compared Decision Tree, Random Forest, Binary Deep Neural Network, Fuzzy Deep Neural Network
6. Saved model weights and wrote code to return JSON of predictions
7. Added rudimentary bill scanner + amount extractor

Future steps:

1. Integrate with web app (serve via Flask?)

#### Data

Upper income dataset - https://www.kaggle.com/datasets/ramyapintchy/personal-finance-data
Lower income dataset - https://www.kaggle.com/datasets/tharunprabu/my-expenses-data

#### Explanation

Read DNN.md for more details
