import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#import the data
data = pd.read_csv("/Users/elliekuang31773/Desktop/US_CO2_CAPITA.csv")
data = data.drop(['States'], axis=1)
data = np.array(data)
data = data.tolist()

#Plot
labels = ['1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990',
'1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
'2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014',
'2015']
y = np.log(data[1])
plt.xlabel('Date')
plt.ylabel('CO2 per Capita (in millions)')
plt.plot(labels, y)
plt.show()

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
