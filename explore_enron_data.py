#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
	
	salary
	to_messages
	deferral_payments
	total_payments
	exercised_stock_options
	bonus
	restricted_stock
	shared_receipt_with_poi
	restricted_stock_deferred
	total_stock_value
	expenses
	loan_advances
	from_messages
	other
	from_this_person_to_poi
	poi
	director_fees
	deferred_income
	long_term_incentive
	email_address
	from_poi_to_this_person
"""

import pickle
import pandas as pd
print "\nLoading Data Set....\n"
enron_data = pickle.load(open("final_project/final_project_dataset.pkl", "r"))
print "Total Number of persons in the enron Dataset:-",len(enron_data)


print "\nAllocation across classes (POI/non-POI)"
print "---------------------------------------"
num_poi=0
num_nonpoi=0
for p in enron_data.values():
	if p['poi']==1:
		num_poi=num_poi+1
	else:
		num_nonpoi=num_nonpoi+1
print "\tNumber of Poi in dataset:-",num_poi
print "\tNumber of Non Poi in dataset:-",num_nonpoi

print "\nNumber of features in the enron dataset:-",len(enron_data.values()[0].keys())

print "\n-------------------------------------------"
print "List of features in the enron dataset:-"
print "-------------------------------------------\n"
fn=[]
for f in enron_data.values()[0].keys():
	print f
	fn.append(f)


print "\n---------------------------------------------------------------------------"
print "Now checking all features to find how many NaN values each feature have"
print "---------------------------------------------------------------------------\n"
nan_data={}
for f in fn:
	c=0
	if f!="email_address" and f!="poi" and f!="deferred_income":
		for p in enron_data:
			if (not (str(enron_data[p][f]).isdigit())):
				c=c+1
		nan_data[f]=pd.Series([c,((float(c)/len(enron_data))*100)],index=["Number of NaN","Percentage NaN"])
	if f=="email_address":
		for p in enron_data:
			if "@" not in enron_data[p]["email_address"]:
				c=c+1
		nan_data[f]=pd.Series([c,((float(c)/len(enron_data))*100)],index=["Number of NaN","Percentage NaN"])
	if f=='poi':
		for p in enron_data:
			if enron_data[p]["poi"] not in [0,1]:
				c=c+1
		nan_data[f]=pd.Series([c,((float(c)/len(enron_data))*100)],index=["Number of NaN","Percentage NaN"])
df=pd.DataFrame(nan_data).transpose()
print df