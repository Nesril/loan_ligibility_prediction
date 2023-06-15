import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import seaborn as sb


data=pd.read_csv("train.csv")

def prepared_data(data):
    data=data.drop(["Loan_ID"],axis=1)
    #print( data.groupby("Gender").agg("count"))
    #gender makes huge differencee therefore don't skip it

    #print( data.groupby("Married").agg("count"))
    #gender does'nt makes huge difference therefore skip it

    data=data.drop(["Married"],axis=1)

    #print( data.groupby("Dependents").agg("count"))
    #gender makes huge differencee therefore don't skip it

    #print( data.groupby("Education").agg("count"))
    #gender makes huge differencee therefore don't skip it

    #print( data.groupby("Self_Employed").agg("count"))
    #gender makes huge differencee therefore don't skip it

    data=data.dropna()
    #print(data.isna().sum())
    #print(data.shape)
    data["Dependents"]=data["Dependents"].apply( lambda x: 3 if x=="3+" else int(x))
    #print(data.Dependents.unique())

    lE_Gen=LabelEncoder()
    data.Gender=lE_Gen.fit_transform(data["Gender"])

    lE_Gen=LabelEncoder()
    data.Education=lE_Gen.fit_transform(data["Education"])

    lE_Gen=LabelEncoder()
    data.Self_Employed=lE_Gen.fit_transform(data["Self_Employed"])

    lE_Gen=LabelEncoder()
    data.Property_Area=lE_Gen.fit_transform(data["Property_Area"])

    lE_Gen=LabelEncoder()
    data.Loan_Status=lE_Gen.fit_transform(data["Loan_Status"])
    #print(data.head())
    return data

data=prepared_data(data)
'''  
   data visualization

#pie chart for loan_status
#pie_loan=data["Loan_Status"].value_counts()
#print(pie_loan)
#plt.pie(pie_loan.values,labels=pie_loan.index,autopct='%1.1f%%')
#plt.show()

lE_Gen=LabelEncoder()
data.Gender=lE_Gen.fit_transform(data["Gender"])
plt.xlabel("Loan_Status")
plt.ylabel("Gender")
plt.hist(data.Gender,rwidth=0.8)
plt.show()


lE_Mar=LabelEncoder()
data.Married=lE_Gen.fit_transform(data["Married"])
plt.hist(data.Married,rwidth=0.8)
plt.xlabel("Married")
plt.ylabel("Count")
plt.show()
print(data.head())
'''

x=data.drop(["Loan_Status"],axis=1)
y=data["Loan_Status"]

'''
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf']
        }  
    },
    "GaussianNB":{ 
         'model':GaussianNB(),
         'params':{
            'var_smoothing':[0.0000001,0.00000001,0.000000001]                 
         }
    },
     "MultinomialNB":{ 
         'model':MultinomialNB(force_alpha=True),
         'params':{
            'alpha':[1.0,0.75,0.50,0.25,0.00],
         }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items(): 
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df)
#from this we can identify which one has higher accuracy: use logistic_regression with c=1
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
l_reg=LogisticRegression(solver='liblinear',multi_class='auto')
model=l_reg.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print("accuracy using same training and testing data: ",accuracy)

test_data=pd.read_csv("train.csv")
test_data=prepared_data(test_data)

x_test=test_data.drop(["Loan_Status"],axis=1)
y_test=test_data["Loan_Status"]

l_reg=LogisticRegression(solver='liblinear',multi_class='auto')
model=l_reg.fit(x,y)
accuracy=model.score(x_test,y_test)
print("accuracy using different training and testing data: ",accuracy)
print("--------------------------------------------------")
