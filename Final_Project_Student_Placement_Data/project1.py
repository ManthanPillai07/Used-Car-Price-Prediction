# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:07:06 2021

@author: Collective 5
"""

# Importing required packages
import pandas as pd
# For Plotting Data
import seaborn as sns
import matplotlib.pyplot as plt
# For implementing the models
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
# Importing the CSV file of Student Placement Data for analysis
stuPlData = pd.read_csv ('student placement data.csv')

# Checking the Top 5 Records
stuPlData.head(5)
# Checking the Bottom 5 Records
stuPlData.tail(5)

# Statistical Summarisaton of the data
stuPl_summary = stuPlData.describe()
print(stuPl_summary)

# Checking no of records in each suggested job role
stuPl_sjrvc=stuPlData['Suggested Job Role'].value_counts()
print(stuPl_sjrvc)

# Checking no of records in each interested career area
stuPl_icavc=stuPlData['interested career area '].value_counts()
print(stuPl_icavc)

# Checking no of records in each certification
stuPl_certvc=stuPlData['certifications'].value_counts()
print(stuPl_certvc)

# Checking no of records in each Workshops
stuPl_wsvc=stuPlData['workshops'].value_counts()
print(stuPl_wsvc)

# Checking no of records in each coding skill ratings
stuPl_csrvc=stuPlData['coding skills rating'].value_counts()
print(stuPl_csrvc)

#finding categorial / descriptive and numeric columns
stuPl_categorial = stuPlData.select_dtypes(include=["object"]).keys()
print(stuPl_categorial)

stuPl_numerics = stuPlData.select_dtypes(include=["int64","float64"]).keys()
print(stuPl_numerics)

# Data Visualisation
#visualizing the hard worker and smart worker
sns.countplot(y=stuPlData['hard/smart worker'])
plt.figure(figsize=(25,25))


#Box Plotting the rating data
plt.boxplot(stuPlData['coding skills rating'], notch=None, vert=None, patch_artist=None, widths=None)


# Creating histogram
sruPl_histo=stuPlData['Overall Percentage']
plt.hist(sruPl_histo,bins= [50,55,60,65,70,75,80,85,90,95,100])
plt.show()

# Creating PieChart to show the data of students interested career area.
pieData=stuPl_icavc.keys()
plt.pie(stuPl_icavc, labels = pieData)

# Cleaning
# using isnull().any() function to check if there are any null values in data
print(stuPlData.isnull().any())
# From the above we can see that there are no NULL data in the dataset so no need
# to remove any data.

# Models are predicting on the basis of numerical data, So we are cleaning the data
# by converting it to caregorical data and then further converting that to numerical data

stuPlData["can work long time before system?"] = stuPlData["can work long time before system?"].astype('category')
stuPlData["self-learning capability?"] = stuPlData["self-learning capability?"].astype('category')
stuPlData["Extra-courses did"] = stuPlData["Extra-courses did"].astype('category')
stuPlData["certifications"] = stuPlData["certifications"].astype('category')
stuPlData["workshops"] = stuPlData["workshops"].astype('category')
stuPlData["talenttests taken?"] = stuPlData["talenttests taken?"].astype('category')
stuPlData["olympiads"] = stuPlData["olympiads"].astype('category')
stuPlData["reading and writing skills"] = stuPlData["reading and writing skills"].astype('category')
stuPlData["memory capability score"] = stuPlData["memory capability score"].astype('category')
stuPlData["Interested subjects"] = stuPlData["Interested subjects"].astype('category')
stuPlData["interested career area "] = stuPlData["interested career area "].astype('category')
stuPlData["Job/Higher Studies?"] = stuPlData["Job/Higher Studies?"].astype('category')
stuPlData["Type of company want to settle in?"] = stuPlData["Type of company want to settle in?"].astype('category')
stuPlData["Taken inputs from seniors or elders"] = stuPlData["Taken inputs from seniors or elders"].astype('category')
stuPlData["interested in games"] = stuPlData["interested in games"].astype('category')
stuPlData["Interested Type of Books"] = stuPlData["Interested Type of Books"].astype('category')
stuPlData["Salary Range Expected"] = stuPlData["Salary Range Expected"].astype('category')
stuPlData["In a Realtionship?"] = stuPlData["In a Realtionship?"].astype('category')
stuPlData["Gentle or Tuff behaviour?"] = stuPlData["Gentle or Tuff behaviour?"].astype('category')
stuPlData["Management or Technical"] = stuPlData["Management or Technical"].astype('category')
stuPlData["Salary/work"] = stuPlData["Salary/work"].astype('category')
stuPlData["hard/smart worker"] = stuPlData["hard/smart worker"].astype('category')
stuPlData["worked in teams ever?"] = stuPlData["worked in teams ever?"].astype('category')
stuPlData["Introvert"] = stuPlData["Introvert"].astype('category')
stuPlData["Suggested Job Role"] = stuPlData["Suggested Job Role"].astype('category')


# saving the suggested job role as a text to map it later on
stuPl_SJR_text = dict(enumerate(stuPlData['Suggested Job Role'].cat.categories))


stuPlData["can work long time before system?"]= stuPlData["can work long time before system?"].cat.codes
stuPlData["self-learning capability?"]= stuPlData["self-learning capability?"].cat.codes
stuPlData["Extra-courses did"]= stuPlData["Extra-courses did"].cat.codes
stuPlData["certifications"]= stuPlData["certifications"].cat.codes
stuPlData["workshops"]= stuPlData["workshops"].cat.codes
stuPlData["talenttests taken?"]= stuPlData["talenttests taken?"].cat.codes
stuPlData["olympiads"]= stuPlData["olympiads"].cat.codes
stuPlData["reading and writing skills"]= stuPlData["reading and writing skills"].cat.codes
stuPlData["memory capability score"]= stuPlData["memory capability score"].cat.codes
stuPlData["Interested subjects"]= stuPlData["Interested subjects"].cat.codes
stuPlData["interested career area "]= stuPlData["interested career area "].cat.codes
stuPlData["Job/Higher Studies?"]= stuPlData["Job/Higher Studies?"].cat.codes
stuPlData["Type of company want to settle in?"]= stuPlData["Type of company want to settle in?"].cat.codes
stuPlData["Taken inputs from seniors or elders"]= stuPlData["Taken inputs from seniors or elders"].cat.codes
stuPlData["interested in games"]= stuPlData["interested in games"].cat.codes
stuPlData["Interested Type of Books"]= stuPlData["Interested Type of Books"].cat.codes
stuPlData["Salary Range Expected"]= stuPlData["Salary Range Expected"].cat.codes
stuPlData["In a Realtionship?"]= stuPlData["In a Realtionship?"].cat.codes
stuPlData["Gentle or Tuff behaviour?"]= stuPlData["Gentle or Tuff behaviour?"].cat.codes
stuPlData["Management or Technical"]= stuPlData["Management or Technical"].cat.codes
stuPlData["Salary/work"]= stuPlData["Salary/work"].cat.codes
stuPlData["hard/smart worker"]= stuPlData["hard/smart worker"].cat.codes
stuPlData["worked in teams ever?"]= stuPlData["worked in teams ever?"].cat.codes
stuPlData["Introvert"]= stuPlData["Introvert"].cat.codes
stuPlData["Suggested Job Role"]= stuPlData["Suggested Job Role"].cat.codes

stuPlData['Suggested Job RoleTEXT'] = stuPlData['Suggested Job Role'].map(stuPl_SJR_text)

# modelling
# select the data to predict
data_to_predict = stuPlData["Suggested Job Role"]
# selecting the data to train the model
data_to_train_names = ['Acedamic percentage in Operating Systems', 'percentage in Algorithms','Percentage in Programming Concepts','Percentage in Software Engineering', 'Percentage in Computer Networks','Percentage in Electronics Subjects','Percentage in Computer Architecture', 'Percentage in Mathematics','Percentage in Communication skills','Hours working per day', 'Logical quotient rating', 'hackathons','coding skills rating', 'public speaking points','can work long time before system?', 'self-learning capability?','Extra-courses did', 'certifications', 'workshops','talenttests taken?', 'olympiads', 'reading and writing skills','memory capability score', 'Interested subjects','interested career area ', 'Job/Higher Studies?','Type of company want to settle in?','Taken inputs from seniors or elders', 'interested in games','Interested Type of Books', 'Salary Range Expected','In a Realtionship?', 'Gentle or Tuff behaviour?','Management or Technical', 'Salary/work', 'hard/smart worker','worked in teams ever?', 'Introvert', 'Suggested Job Role']

#data_to_train_names = ['can work long time before system?', 'self-learning capability?','Extra-courses did', 'certifications', 'workshops','talenttests taken?', 'olympiads', 'reading and writing skills','memory capability score', 'Interested subjects','interested career area ', 'Job/Higher Studies?','Type of company want to settle in?','Taken inputs from seniors or elders', 'interested in games','Interested Type of Books', 'Salary Range Expected','In a Realtionship?', 'Gentle or Tuff behaviour?','Management or Technical', 'Salary/work', 'hard/smart worker','worked in teams ever?', 'Introvert', 'Suggested Job Role']
    
#data_to_train_names = ['Acedamic percentage in Operating Systems', 'percentage in Algorithms', 'Percentage in Programming Concepts', 'Percentage in Software Engineering', 'Percentage in Computer Networks','Percentage in Electronics Subjects','Percentage in Computer Architecture','Percentage in Mathematics','Percentage in Communication skills','Hours working per day','interested career area ','hard/smart worker']

data_to_train = stuPlData[data_to_train_names]

data_to_train.describe()

# Decision Tree Regressor with 100% data to check if the model is underfitting or not
# model 1
student_model = DecisionTreeRegressor(random_state=1)
student_model.fit(data_to_train, data_to_predict)
print("Making predictions for the following 5 houses:")
print(data_to_train.head())
print("The predictions are")
print(student_model.predict(data_to_train.head()))
predected_job_role = student_model.predict(data_to_train)
mean_absolute_error(data_to_predict, predected_job_role)

# Decision Tree Regressor 30% for test size and 70% for trainng
train_X, val_X, train_y, val_y = train_test_split(data_to_train, data_to_predict, test_size=0.3, random_state = 0)
# Definemodel 2
student_model_reg = DecisionTreeRegressor(random_state=1)
# Fit model
student_model_reg.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = student_model_reg.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
print(val_predictions)

# random forest model predicting for 30% on the basis of 70% data
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
forest_model.predict(val_X)

# KNeighborsClassifier Model 30% on the basis of 70% data
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(train_X, train_y)
y_pred = knn.predict(val_X)
print(mean_absolute_error(val_y, y_pred))