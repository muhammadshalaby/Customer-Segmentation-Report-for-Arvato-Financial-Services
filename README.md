# Customer Segmentation Report for Arvato Financial Services by Mohamed Shalaby 30/08/2022

In this project we tried to cluster and predict future customers from general population datasets.

We have analyzed four datasets in this project:
1- `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
2- `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
3- `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
4- `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
- In addition to helper file "function.py" to clean our datasets and evaluate ML algorithms.

## The project design is:

1- Data assessment and cleaning
2- Data Visualization
3- Feature Engineering
4- Unsupervised Machine learning algorithm (KMeans)
5- Supervised Machine Learning (Classification)

## Requirements
The whole project is written in jupyter notebook with the following libraries:

python 3.9.13 | packaged by conda-forge

library    version
------------------
numpy       1.23.2
pandas      1.4.3
matplotlib  3.5.3
sklearn     1.1.2
autogluon   0.5.1
