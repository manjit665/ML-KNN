from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib

cancer=load_breast_cancer()
#print(cancer.keys())
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)
neighbor=[1,3,5,7,9,11,14,16]
train_acc=[]
test_acc=[]


for i in neighbor:
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    train_acc.append(model.score(X_train,y_train))
    test_acc.append(model.score(X_test,y_test))
for i in range(len(train_acc)):
    print("neighbors:",neighbor[i],"\ttrain:",train_acc[i] ,"\ttest:", test_acc[i])

model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train,y_train)
joblib.dump(model,"KNN.pkl")