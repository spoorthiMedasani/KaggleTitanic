########################################################################################################
########################################################################################################
###################                                                                  ###################
###################                     VARIABLE TRANSFORMATION                      ###################
###################                                                                  ###################
########################################################################################################
########################################################################################################

import pandas as pd
import numpy as np

train= pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
#print(train.head())

fulldata= train.append(test)
#print(train.shape)
#print(test.shape)
#print(fulldata.shape)

#print(fulldata.head())
#print(fulldata.dtypes)

fulldata['Age']=fulldata['Age'].fillna(np.mean(fulldata['Age']))
fulldata['Age']=fulldata['Age'].astype(int)
fulldata['Cabin']= fulldata['Cabin'].astype(str).str[0]
#print(fulldata.head())
#print(fulldata.groupby(['Cabin'])['Cabin'].count())
dummies= pd.get_dummies(fulldata['Cabin'],prefix='Cabin')
fulldata= pd.concat([fulldata,dummies],axis=1)


#print(fulldata.groupby(['Embarked'])['Embarked'].count())
dummies2=pd.get_dummies(fulldata['Embarked'],prefix='Emb')
fulldata=pd.concat([fulldata,dummies2],axis=1)


fulldata['Fare']=fulldata['Fare'].fillna(np.mean(fulldata['Fare']))

fulldata['Family']=fulldata['Parch']+fulldata['SibSp']+1

dummies3=pd.get_dummies(fulldata['Pclass'],prefix='Class')
fulldata=pd.concat([fulldata,dummies3],axis=1)


dummies4=pd.get_dummies(fulldata['Sex'])
fulldata=pd.concat([fulldata,dummies4],axis=1)



def title(name):
    z=name.find('.')
    z1=name.find(',')
    return (name[z1+1:z].strip())

def title2(title):
    if title=='Master':
        return('Master')
    elif title=='Miss' or title=='Mlle':
        return('Miss')
    elif title=='Mr':
        return('Mr')
    elif title=='Mrs':
        return('Mrs')
    else:
        return('Misc')
    
    
fulldata['Title']=fulldata['Name'].apply(title)
#print(fulldata.groupby(['Title'])['Title'].count())
fulldata['Title2']=fulldata['Title'].apply(title2)
#print(fulldata.groupby(['Title2'])['Title2'].count())

dummies5= pd.get_dummies(fulldata['Title2'],prefix='Title')
fulldata= pd.concat([fulldata,dummies5],axis=1)



def age(age):
    if age in range(0,20):
        return('0_20')
    elif age in range(21,40):
        return('21_40')
    elif age in range(41,60):
        return('41_60')
    else:
        return('>60')
    
fulldata['Age2']=fulldata['Age'].apply(age)
dummies6= pd.get_dummies(fulldata['Age2'],prefix='Age')
fulldata=pd.concat([fulldata,dummies6],axis=1)


train=fulldata[0:891]
test=fulldata[891:]
#print(train.shape)
#print(test.shape)
#print(train.isnull().sum())
#print(test.isnull().sum())

survived=train['Survived'].astype(int)
del train['Survived']
train=pd.concat([train,survived],axis=1)
z=train.isnull().sum()
#z.to_csv('Vars.csv')

print(train.head())
#del fulldata['Cabin']
#del fulldata['Embarked']
#del fulldata['Pclass']
#del fulldata['Sex']
del (fulldata['Title'],fulldata['Name'])
#del (fulldata['Title2'])
del (fulldata['Ticket'])
#del (fulldata['Age2'])

########################################################################################################
########################################################################################################
###################                                                                  ###################
###################                          DATA EXPLORATION                        ###################
###################                                                                  ###################
########################################################################################################
########################################################################################################

import matplotlib.pyplot as plt
#import seaborn as sns

#p1=sns.factorplot(x="male",y="Survived",data=train,kind="bar",size=6,palette="muted")
#p1.despine(left=True)
#p1=p1.set_ylabels("survival prop")
#p2=sns.factorplot(x="Family",y="Survived",data=train,kind="bar",size=6,palette="muted")
#p2.despine(left=True)
#p2=p2.set_ylabels("survival prop")
#plt.savefig('Age.png')

def plots(var):
    z=train.groupby([var])['Survived'].mean()
    plt1=z.plot.bar()
    plt.xlabel(var,fontsize=18)
    plt.ylabel('Survived',fontsize=18)
    plot_title= var[:]
    plot_title=(plot_title+".png")
    plt.savefig(plot_title)
    return()

variables=['Age'
           ,'Cabin'
           ,'Embarked'
           ,'Fare'
           ,'Parch'
           ,'Pclass'
           ,'Sex'
           ,'SibSp'
           ,'Family'
           ,'Age2']

#for i in variables:
#    plots(i)

def box_plot(var2):
    plt2=plt.boxplot(train[var2])
    plot_title= (var2+"_box"+".png")
    plt.savefig(plot_title)
    return()

continuous_vars=['Age'
                 ,'Fare'
                 ,'SibSp'
                 ,'Parch']

#for i in continuous_vars:
#    box_plot(i)

print(train.shape)
del (train['Cabin'],train['Embarked'],train['Name'],train['PassengerId'],
     train['Pclass'],train['Sex'],train['Ticket'],train['Title'],train['Title2'],
     train['Age2'])
del (test['Cabin'],test['Embarked'],test['Name'],test['PassengerId'],
     test['Pclass'],test['Sex'],test['Ticket'],test['Title'],test['Title2'],
     test['Age2'])
print(train.shape)

########################################################################################################
########################################################################################################
###################                                                                  ###################
###################                          CORRELATION MATRIX                      ###################
###################                                                                  ###################
########################################################################################################
########################################################################################################

#plt.matshow(train.corr())
#plt.savefig('correlationmatrix.png')

#z=train.corr()
#z.to_csv('Correlation_matrix.csv')





########################################################################################################
########################################################################################################
###################                                                                  ###################
###################                          DATA MODELING                           ###################
###################                                                                  ###################
########################################################################################################
########################################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn import metrics

y= train['Survived']
del (train['Survived'])

print(train.shape)

RF= RandomForestClassifier(max_depth=5, n_estimators=200)
RF.fit(train,y)

y_pred= RF.predict(train)
score= metrics.f1_score(y, y_pred)
pvalues= chi2(train,y_pred)
print(pvalues)
print(score)
z=(y,y_pred)
np.savetxt('predicted.csv',z,delimiter=',')

print(test.shape)
del (test['Survived'])

test_pred= RF.predict(test)
np.savetxt('test_pred.csv',test_pred,delimiter=',')


##from sklearn.ensemble import ExtraTreesClassifier
##
##ET= ExtraTreesClassifier(max_depth=8)
##ET.fit(train,y)
##y_pred= ET.predict(train)
##score=metrics.f1_score(y,y_pred)
##
##print(score)
##
##test_pred= ET.predict(test)
##np.savetxt('test_pred.csv',test_pred,delimiter=',')



##from sklearn.ensemble import AdaBoostClassifier
##
##ADA= AdaBoostClassifier(n_estimators=100,learning_rate=0.75)
##ADA.fit(train,y)
##y_pred= ADA.predict(train)
##score=metrics.f1_score(y,y_pred)
##
##print(score)
##
##test_pred= ADA.predict(test)
##np.savetxt('test_pred.csv',test_pred,delimiter=',')
