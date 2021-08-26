#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

# How many data points (people) are in the dataset?
print (len(enron_data))
print (len(enron_data.keys()))

# For each person, how many features are available?
print (len(enron_data['SKILLING JEFFREY K']))

# How many POIs are there in the E+F dataset?
#print (enron_data['SKILLING JEFFREY K']['poi'])
count = 0 
for user in enron_data:
    if enron_data[user]['poi'] == 1:
        count += 1
print (count)
#print (enron_data['SKILLING JEFFREY K'])

poi = [x for x, y in enron_data.items() if y['poi']]
#print(poi)
print(len(poi))

#How many POI’s were there total?
poi_text = '../final_project/poi_names.txt'
poi_names = open(poi_text, "r")
fr = poi_names.readlines()
#print(fr)
print(len(fr[2:]))
poi_names.close()

#What is the total value of the stock belonging to James Prentice?
print(enron_data['PRENTICE JAMES'])
print(enron_data['COLWELL WESLEY'])
print(enron_data['SKILLING JEFFREY K'])

#How much money did that person get?
names = ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']
names_payments = {name:enron_data[name]['total_payments'] for name in names}
print (sorted(names_payments.items(), key=lambda x: x[1], reverse=True))

#How many folks in this dataset have a quantified salary? What about a known email address?
import pandas as pd
df = pd.DataFrame(enron_data)
#print(df)
print(sum(df.loc['salary',:]!='NaN'))
print(sum(df.loc['email_address',:] != 'NaN'))

