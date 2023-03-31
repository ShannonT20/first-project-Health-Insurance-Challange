import streamlit as st
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import time
data_health=pd.read_csv("C:/Users/hp/Videos/Dynasty/health_insurance.csv")
def normalise(x):
        min_max_scaler=preprocessing.MinMaxScaler().fit_transform(x)
        return min_max_scaler
    
#label encoding for smoker and region
data_health.replace({"sex":{'female':0,'male':1}},inplace=True)
data_health.replace({"smoker":{'no':0,'yes':1}},inplace=True)
data_health.replace({"region":{'southwest':0,'southeast':1,'northwest':2,'northeast':3}},inplace=True)


#Data seperation into Feature and Target 
X=data_health[["age","sex","bmi","children","smoker","region","salary"]].values
Y=data_health["insured"].values

X=normalise(X)

#Split the data into train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

#Fitting decision classifier to the model
from sklearn.tree import DecisionTreeClassifier  
data_health_classifier= DecisionTreeClassifier(criterion='entropy',random_state=0)  
data_health_classifier.fit(X_train, Y_train)  

#Predicting the test set result  
Y_pred=data_health_classifier.predict(X_test)  
#New values in variable A

def health_prediction(input_data):
    X_1=data_health[["age","sex","bmi","children","smoker","region","salary"]].values
    X_1=np.vstack([X_1,input_data])#append it to array
    
    X_1=normalise(X_1)
    
    #Prediciting using Decision Tree Classifier
    New_Pred=data_health_classifier.predict(X_1[-1:]) 
    if 0 in New_Pred:
        return "This Person CAN NOT Insured!☹🙁"
    else:
        return "This Person CAN Be Insured!👍✔"

def main():
    #giving title
    st.title("Health Insurance Web Application")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.write("")
    with col2:
        image = Image.open("C:/Users/hp/Videos/Dynasty/health_in.jpg",)
        image=image.resize((120,150))
        st.image(image)
    with col3:
        st.write("")
    # You can always call this function where ever you want
    #def add_logo(logo_path, width, height):
     #   #Read and return a resized logo
      #  logo = Image.open(logo_path)
       # modified_logo = logo.resize((width, height))
       # return modified_logo

    #st.sidebar.image(add_logo(logo_path="C:/Users/hp/Videos/Dynasty/health_in.jpg", width=50, height=60))
    
    age=st.text_input("Age")
    sex=st.radio(label="Sex",options=["Male","Female"])
    if sex=="Male":
        sex=1
    elif sex=="Female" :
        sex=0
    bmi=st.slider("BMI",10,50)
    children=st.text_input("Children")
    smoker=st.radio(label="Smoker",options=["Yes","No"])
    if smoker=="Yes":
        smoker=1
    elif smoker=="No":
        smoker=0
    Region=st.radio(label="Region" ,options=["Southwest","Southeast","Northwest","Northeast"])
    if Region=="Southwest":
        Region=0
    elif Region=="Southeast":
        Region=1
    elif Region=="Northwest":
        Region=2
    elif Region=="Northwest":
        Region=3
    salary=st.text_input("Salary")
    
    #code for prediction
    health_pd = ''

    #creating button for prediction
    if st.button("Check Status"):
       health_pd = health_prediction([age,sex,bmi,children,smoker,Region,salary])
       with st.spinner("Wait..."):
            time.sleep(4)
    st.success(health_pd)
if __name__=='__main__':
    main()        

  