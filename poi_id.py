#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from tester import test_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

features = ["salary", "bonus"]

# data = featureFormat(data_dict, features)
# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     plt.scatter( salary, bonus )

# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()


del data_dict['TOTAL']

# data = featureFormat(data_dict, features)
# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     plt.scatter( salary, bonus )

# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()



### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Two new feature adding fraction_from_poi,fraction_to_poi
for item in my_dataset:
	person = my_dataset[item]
	if (all([	person['from_poi_to_this_person'] != 'NaN',
				person['from_this_person_to_poi'] != 'NaN',
				person['to_messages'] != 'NaN',
				person['from_messages'] != 'NaN'
			])):
	    person["fraction_from_poi"] = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
	    person["fraction_to_poi"] = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
	else:
	    person["fraction_from_poi"] = person["fraction_to_poi"] = 0

# First lets check the score without the new features.
# features_list=features_list+['salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock',
# 'shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees',
# 'deferred_income','long_term_incentive','from_poi_to_this_person']

# Now include the new features to see the affect on the score.
features_list=features_list+['fraction_to_poi','fraction_from_poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock',
'shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees',
'deferred_income','long_term_incentive','from_poi_to_this_person']
### Extract features and labels from dataset for local testing

#List of features along with the k score
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
k_best = SelectKBest()
k_best.fit(features, labels)
results_list = zip(features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[1], reverse=True)
# print results_list

# Plot to decide how many features to choose
# fa=[]
# va=[]
# for f,v in results_list:
# 	fa.append(f)
# 	va.append(v)
# y_pos = np.arange(len(fa))
# plt.bar(y_pos,va,align='center')
# plt.xticks(y_pos,fa,rotation=90)
# plt.xlabel("Features")
# plt.ylabel("K_Score")
# plt.show()

# print "K-best features:", results_list

# final fearure list with five features from result list. Because the first five features has high k score and then after 5th feature k score decreases more.
features_list=['poi']
for i in range(5):
	features_list.append(results_list[i][0])
# print features_list



### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels,test_size=.3,random_state=42)
cv = StratifiedShuffleSplit(labels,n_iter = 100, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
# Provided to give you a starting point. Try a variety of classifiers.
# print cv


print "\nGausianNB"
from sklearn.naive_bayes import GaussianNB
gc = GaussianNB()
gc.fit(features_train,labels_train)
test_classifier(gc,my_dataset,features_list)

print "\nDecisionTree"
from sklearn import tree
dc=tree.DecisionTreeClassifier()

param_grid ={'criterion':['entropy','gini']}
dc=GridSearchCV(dc,param_grid =param_grid)
dc.fit(features_test,labels_test)
test_classifier(dc.best_estimator_,my_dataset,features_list)
# print "Accuracy Score",accuracy_score(labels_test,dc.predict(features_test))
# print "Percision Score",precision_score(labels_test,dc.predict(features_test))
# print "Recall Score",recall_score(labels_test,dc.predict(features_test))


# Feature Scaling
scale=MinMaxScaler()
scale.fit(features_train)
features_train=scale.transform(features_train)


print "\nKNN"
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(weights='distance')

param_grid ={'n_neighbors':[4,5,6]}
kn=GridSearchCV(estimator=kn,param_grid =param_grid)
kn.fit(features_train,labels_train)

test_classifier(kn.best_estimator_,my_dataset,features_list)
# print "Accuracy Score",accuracy_score(labels_test,kn.predict(features_test))
# print "Precision Score",precision_score(labels_test,kn.predict(features_test))
# print "Recall Score",recall_score(labels_test,kn.predict(features_test))



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# param_grid ={'criterion':['gini','entropy']}
# clf=GridSearchCV(dc,param_grid =param_grid)

clf=kn.best_estimator_

# summarize the results of the grid search
# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
