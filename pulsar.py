import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier

pulsar_path = 'pulsar_stars.csv'
pulsar_data = pd.read_csv(pulsar_path)
#pulsar_data.describe() Let's us see the data descriptions for pulsar data
#Check for missing data we find none:
#pulsar_data.isnul().sum()
# This is the output of pulsar_data.columns
# Index([' Mean of the integrated profile',
#        ' Standard deviation of the integrated profile',
#        ' Excess kurtosis of the integrated profile',
#        ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',
#        ' Standard deviation of the DM-SNR curve',
#        ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve',
#        'target_class'],
#       dtype='object')
pulsar_features = [' Mean of the integrated profile', \
       ' Standard deviation of the integrated profile', \
       ' Excess kurtosis of the integrated profile', \
       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve', \
       ' Standard deviation of the DM-SNR curve', \
       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']
X = pulsar_data[pulsar_features]
y= pulsar_data['target_class']
pulsar_model = DecisionTreeClassifier(random_state=1)
pulsar_model.fit(X,y)
z =  pulsar_model.predict(X)

print(pulsar_model.get_n_leaves())
#This is what we will be using to get a confusion matrix
pd.crosstab(pd.Series(y,name="Actual"),pd.Series(z,name="Predicted"))