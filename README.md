# Identify-Fraud-from-Enron-Email

## Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will play detective, and put my new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist me in my detective work, Udacity have combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity

## About Data Set:- 
This data set contains all emails conversations between the top persons in the company those may be involved in the fraud. This dataset also contains all finacial information like salary,bonous etc.
The data set contains 146 person.
The persons in the data set can be classified into two two categories POI and Non POI. POI refers to person of interest. All the members which are expected to be involved in the fraud and also the members of high post are included in the POI and all other fall into Non Poi list.
The data set contains a feature named as poi which has two values 0 or 1. If the person belong to the poi list then the value of poi will be 1 otherwise 0.
So i tried to find how many POI person are there in the dataset. And for that i just counted all the 1's in the dataset poi feature.
The number of poi person are:- 18
So it means there are 18 members of poi in the dataset.
Now as there is one feature poi there are 21 features in the dataset including financial features.
The list can be found by running the python script named explore_enron_data.py
One more thing that is important in the dataset exploration is that we should know how much data we have and how much not. So in order to find that i tried to find the number of missing values or NaN values in the features of the dataset along with the percentage showing the percentage of data in the feature.
The feature with the minimum missing value is poi and this feature has no missing value.
The feature with the maximum missing value is restricted_stock_deferred and 98% values are missing in this feature.
Bonous feature contains 43% NaN values, Salary feature contains 34.9% NaN values it means we don't have salary for 51 person out of 146, We don't have email address for 35 person in the data set of 146.

## Goal And Machine learning benefits:- 
My Goal in this project is to identify correctly the person's those may have made the fraud in the company i.e poi(person of interest).And for this i have to analyse the emails between the poi and non poi and their finacial.And Machine learning can be helpful because machine can analyse the training data and on the basis of that data it will analyse it and can be used to predict the data to find the fraud. Machine learning can make the task a lot easier than doing without machine learning. Humans can not analyse large data and can not find distinct pattern in the dataset because the dataset is very large.But machines can do this task in a better way.
