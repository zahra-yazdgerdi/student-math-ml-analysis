import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("D:/ML/mine/2/student-mat.csv") 
#print(df.describe())
#print(df.info())
#print(df.head())

#####clearing data 
#print(df.isna().sum())
#print(df.isnull().any(axis=1))
###in a case that any data is not a number and its like a sign (!, -- , ?, missing) you can use this code
#df['']=pd.to_numeric(df[''],errors='coerce)

#####renaming the columns
#print(df.dtypes)
df.columns = ['school','sex','age','address','family_size','parents_status','mother_education','father_education',
           'mother_job','father_job','reason','guardian','commute_time','study_time','failures','school_support',
          'family_support','paid_classes','activities','nursery','desire_higher_edu','internet','romantic','family_quality',
          'free_time','go_out','weekday_alcohol_usage','weekend_alcohol_usage','health','absences','period1_score','period2_score','final_score']

####drawing the heatmap helps us to determine the relationship between the features and identify the cucial ones
#making the corelation table
numeric_df = df.select_dtypes(include=['int64'])
corr = numeric_df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="Reds")
plt.title('Correlation Heatmap', fontsize=20)
#plt.show()

#####Encoding (you can use lablencoder as well, but its more accurate and for the decision tree and others, its better)
df_encode=pd.get_dummies(df, drop_first=True)
print(df_encode.head(5))
#print(df_encode.dtypes)
#df_encode is not included final grade yet !

##### convert final_score to categorical variable # Good:15~20 Fair:10~14 Poor:0~9 and name it by final_grade
df['final_grade'] = 'na'
df.loc[(df.final_score >= 15) & (df.final_score <= 20), 'final_grade'] = 'good' 
df.loc[(df.final_score >= 10) & (df.final_score <= 14), 'final_grade'] = 'fair' 
df.loc[(df.final_score >= 0) & (df.final_score <= 9), 'final_grade'] = 'poor' 
#final grade is added to df not df_encode 

#####Encoding of final grade, why not use dummies?: as dummies make a new columns for each part , its hard to follow
la=LabelEncoder()
df.final_grade=la.fit_transform(df.final_grade)

#print(df_encode.dtypes)

#####Modeling 
x=df_encode.copy()        # .copy is to avoid changing on main df_encode
y=df['final_grade']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#######################################################################################################

###### KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
ac=accuracy_score(y_test,y_pred)
print("Accuracy knn:",ac)

#######################################################################################################

##### Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
tree=DecisionTreeClassifier( max_depth=5,min_samples_leaf=17,random_state=42,criterion='entropy')
tree.fit(x_train,y_train)
y_tree_predict=tree.predict(x_test)
ac_tree=accuracy_score(y_test,y_tree_predict)
print("Acuuracy_tree:",ac_tree)
scores = cross_val_score(tree, x, y, cv=5)
print("Cross Validation Score_ tree :",scores , "Average CV accuracy_tree:", scores.mean())

###confusion metrix 
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
ac_tree_2=accuracy_score(y_tree_predict,y_test)

cm=confusion_matrix(y_tree_predict,y_test)
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['class=0','class=1','class=2'])
cm_display.plot(cmap=plt.cm.Reds)
#plt.show()

report=classification_report(y_test, y_tree_predict)
print(report)

#######################################################################################################

######Random forest
from sklearn.ensemble import RandomForestClassifier
Random=RandomForestClassifier()
R=Random.fit(x_train,y_train)
y_random_predict=Random.predict(x_test)

ac_random=accuracy_score(y_random_predict,y_test) 
print("Accuracy Random:",ac_random)    

print("Forest Model Score",Random.score(x_train, y_train) , "," ,
      "Forest Cross Validation Score:", Random.score(x_test, y_test))

report=classification_report(y_test,y_random_predict)
print("report random:",report)

cm_random=confusion_matrix(y_random_predict,y_test)
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm_random,display_labels=['class=0','class=1','class=2'])
cm_display.plot(cmap=plt.cm.Blues)
#plt.show()

feature_importance=Random.feature_importances_
features=features = x.columns.tolist()
#print(features)

## drawing the importance of features 
plt.figure(figsize=(10,6))
plt.barh(features,feature_importance,color='blue')
plt.xlabel('feature importance')
plt.ylabel('feature')
#plt.show()

#######################################################################################################

####SVM
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_svc_predict=svc.predict(x_test)
ac_svc=accuracy_score(y_test,y_svc_predict)
print("Acuracy svc:",ac_svc)
print("SVC Model Score",svc.score(x_train, y_train) , "," ,
      "Cross Validation Score:", svc.score(x_test, y_test))


###### new confusion metrix code
cm=confusion_matrix(y_test,y_svc_predict)
print(cm)
df_cm = pd.DataFrame(cm, range(3),
                  range(3))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})
#plt.show()

report_svc = classification_report(y_test, y_svc_predict)
print("report svc:",report)
#######################################################################################################
