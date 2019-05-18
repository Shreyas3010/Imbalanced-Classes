import pandas as pd
import sys
import xgboost
from sklearn.ensemble import RandomForestRegressor
import collections
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def sortSecond(val): 
    return val[1] 


# =============================================================================
#row2=np.arange(5)
#kneighbors= pd.DataFrame(data=None,index=row2,columns = [ 'Index','Dist1','Class'])
# def settointial():
#     for i in range(5):
#         kneighbors['Class'][i]=-1
#         kneighbors['dist1'][i]=sys.float_info.max
#     return
# =============================================================================


kneighbors=[]


def knn1(data1,k):
    in1=0
    data1_cols=data1.columns
    for i1 in range(len(data1)):
        sum1=0.0
        if i1!=k:
              for cl1 in range(len(data1_cols)-2):
                  sum1=sum1+math.pow((data1[data1_cols[cl1+1]][k]-data1[data1_cols[cl1+1]][i1]),2)     
              if in1!=5:
                  kneighbors.append((data1['index'][i1],math.sqrt(sum1),data1['Class'][k]))
                  #kneighbors['Class'][in1]=data1['Class'][k]
                  #kneighbors['Index'][in1]=data1['index'][i1]
                  #kneighbors['Dist1'][in1]=math.sqrt(sum1)
                  in1=in1+1
              else:
                  if math.sqrt(sum1)<kneighbors[4][1]:
                      kneighbors.pop()
                      kneighbors.append((data1['index'][i1],math.sqrt(sum1),data1['Class'][k]))
                      #kneighbors['Index'][4]=data1['index'][i1]
                      #kneighbors['Dist1'][4]=math.sqrt(sum1)
                      #kneighbors.sort(key=sortSecond)
              kneighbors.sort(key=sortSecond) 
    return    
        
        
row1=np.arange(3)
results= pd.DataFrame(data=None,index=row1,columns = [ 'Class','Datasize','Training Datasize','After Sampling','Testing Datasize'])



a1=0
data= pd.read_excel (r'New_Data_BF-2_Raigad.xlsx')
data=data.drop(data.index[0])
data=data[data.Outlier!='Yes']
data=data[data!='ppp']
data=data.dropna(axis=0)
l1=[]
for i in np.arange(len(data)):
    l1.append(2)
data['Class']=l1
data.rename(columns={'Total.Production.(mt)':'Total_Production'}, inplace=True)
data.loc[data['Total_Production'] < 3200, 'Class'] = 0
data.loc[data['Total_Production'] > 4200, 'Class'] = 1
#data['F/C.Top.Pressure.(Kg/cm2)'] = data.F/C.Top.Pressure.(Kg/cm2).astype(float64)
#m=data.columns[data.isna().any()].tolist()
data= data.drop(['Outlier'],axis=1)
#data= data.drop(['index'],axis=1)
data= data.drop(['F/C.Top.Pressure.(Kg/cm2)'],axis=1)
X = data.loc[:, data.columns != 'Total_Production']
y = data.loc[:, data.columns == 'Total_Production']
datasize=collections.Counter(X['Class'])
print("data size",datasize)
results['Class'][a1]=a1
results['Class'][a1+1]=a1+1
results['Class'][a1+2]=a1+2
results['Datasize'][a1]=datasize[a1]
results['Datasize'][a1+1]=datasize[a1+1]
results['Datasize'][a1+2]=datasize[a1+2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,train_size=0.8, random_state = 0)
date_test= X_test.loc[:, X_test.columns == 'Date']
X_train= X_train.drop(['Date'],axis=1)
X_test= X_test.drop(['Date'],axis=1)

trainingdatasize=collections.Counter(X_train['Class'])
testingdatasize=collections.Counter(X_test['Class'])
print("training data size",trainingdatasize)
print("testing data size",testingdatasize)

results['Training Datasize'][a1]=trainingdatasize[a1]
results['Training Datasize'][a1+1]=trainingdatasize[a1+1]
results['Training Datasize'][a1+2]=trainingdatasize[a1+2]

results['Testing Datasize'][a1]=testingdatasize[a1]
results['Testing Datasize'][a1+1]=testingdatasize[a1+1]
results['Testing Datasize'][a1+2]=testingdatasize[a1+2]
X_train_cols=X_train.columns
y_train_cols=y_train.columns

row3=np.arange(1)
tmp_x_train_tuple=()
for list_num in range(len(X_train_cols)):
    tmp_x_train_tuple=tmp_x_train_tuple+(1.1,)
tmp_x_train_list=[tmp_x_train_tuple]
    
tmp_x_train= pd.DataFrame(data=tmp_x_train_list,index=row3,columns = X_train_cols)
tmp_y_train= pd.DataFrame(data=[1.2],index=row3,columns = y_train_cols)
x1_train=X_train[X_train.Class==1]
x0_train=X_train[X_train.Class==0]
x1_train=x1_train.reset_index()
x0_train=x0_train.reset_index()


for x0_train_rownum in range(len(x0_train)):
    kneighbors.clear()
    knn1(x0_train,x0_train_rownum)
    index_2=kneighbors[0][0]
    random_val1=random.uniform(0,1)
    random_val2=1-random_val1
    index_1=x0_train['index'][x0_train_rownum]
    for colnum in range(len(X_train_cols)-1):
        tmp_x_train[X_train_cols[colnum]][0]=((random_val2*X_train[X_train_cols[colnum]][index_1])+(random_val1*X_train[X_train_cols[colnum]][index_2]))/(random_val1+random_val2)
    tmp_x_train['Class'][0]=0
    X_train=X_train.append(tmp_x_train)
    tmp_y_train[y_train_cols[0]][0]=int(((random_val2*y_train[y_train_cols[0]][index_1])+(random_val1*y_train[y_train_cols[0]][index_2]))/(random_val1+random_val2))
    y_train=y_train.append(tmp_y_train)    

for x1_train_rownum in range(len(x1_train)):
    kneighbors.clear()
    knn1(x1_train,x1_train_rownum)
    index_2=kneighbors[0][0]
    random_val1=random.uniform(0,1)
    random_val2=1-random_val1
    index_1=x1_train['index'][x1_train_rownum]
    for colnum in range(len(X_train_cols)-1):
        tmp_x_train[X_train_cols[colnum]][0]=((random_val2*X_train[X_train_cols[colnum]][index_1])+(random_val1*X_train[X_train_cols[colnum]][index_2]))/(random_val1+random_val2)
    tmp_x_train['Class'][0]=1
    X_train=X_train.append(tmp_x_train)           
    tmp_y_train[y_train_cols[0]][0]=int((random_val2*y_train[y_train_cols[0]][index_1])+(random_val1*y_train[y_train_cols[0]][index_2])/(random_val1+random_val2))
    y_train=y_train.append(tmp_y_train)
    
samplingdatasize=collections.Counter(X_train['Class'])
print("samplinging data size",samplingdatasize)

results['After Sampling'][a1]=samplingdatasize[a1]
results['After Sampling'][a1+1]=samplingdatasize[a1+1]
results['After Sampling'][a1+2]=samplingdatasize[a1+2]

#remove class field
X_train= X_train.drop(['Class'],axis=1)
X_test= X_test.drop(['Class'],axis=1)

#classifier

#clf = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
#clf.fit(X_train,y_train)
clf=RandomForestRegressor(n_estimators=100,oob_score=True)
clf.fit(X_train,y_train.values.ravel())
y_pred=clf.predict(X_test)

x1=date_test['Date']
y1=y_test['Total_Production']
y_test_arr=np.array(y1)
num_test=len(y_pred)
y2=y_pred
diff1=np.arange(num_test,dtype=np.float)
sum1=0
for i in range(num_test):
    diff1[i]=abs(y_pred[i]-y_test_arr[i])/y_pred[i]
    sum1=sum1+diff1[i]
   
min1=min(diff1)
max1=max(diff1)
print("min",min1)
print("max",max1)
avg1=sum1/num_test
print("avg",sum1/num_test)

maxstr=str(max1)
minstr=str(min1)
avgstr=str(avg1)
plt.figure(figsize=(20,10))
plt.suptitle('Min : '+minstr+'Max : '+maxstr+'Avg : '+avgstr, fontsize=14, fontweight='bold')
plt.plot(x1,diff1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="Actual Data")
plt.title('Performance (XGB)')
plt.xticks(rotation=90)
plt.xlabel('Production')
plt.ylabel('(Predicted-Actual)/Predicted (%)')
plt.savefig('resultsxgbsmoteregressionnewdata.png')
#plt.savefig('resultsrfsmoteregressionnewdata.png')
plt.show()





#plot

plt.figure(figsize=(20,10))
plt.plot(x1,y1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="Actual Data")
plt.plot(x1,y2,marker='o', markerfacecolor='forestgreen', markersize=7, color='lightgreen', linewidth=0,label="Predicted Data")
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Production')
plt.title('Actual Data vs Predicted Data(XGB)')
plt.legend()
plt.savefig('xgbsmoteregressionnewdata.png')
#plt.savefig('rfsmoteregressionnewdata.png')
plt.show()

    


# plot feature importance

plt.figure(figsize=(20,10))
train_cols=X_train.columns.values
print(train_cols)
print(clf.feature_importances_)
plt.title('Feature importances (XGB)')
plt.bar(train_cols, clf.feature_importances_)
plt.xticks(rotation=90)
plt.ylabel('Feature Importance (%)')
plt.xlabel('Features')
plt.legend()
plt.savefig('xgbsmoteregressionfeatureimportancenewdata.png')
#plt.savefig('rfsmoteregressionfeatureimportancenewdata.png')
plt.show()
