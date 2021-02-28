from keras.models import load_model
import pandas as pd
import numpy as np
import pickle

def predictions(x):
    labels={"Low":0, "Medium":1, "High":2,'white':10,'red':11}
    mdl = load_model("qualitymodel.h5")
    model=pickle.load(open('typemodel.sav','rb'))
    quality_pred=np.argmax(mdl.predict(x),axis=-1)
    type_pred=model.predict(x)
    for k,v in labels.items():
        if v==quality_pred  :
            quality=k
        if type_pred==v:
            wine=k
    return [wine,quality]
def ip():
    features=['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']
    i1=float(input("Enter fixed acidity content  :"))
    i2=float(input("Enter volatile acidity content :"))
    i3=float(input("Enter citric acid content :"))
    i4=float(input("Enter residual sugar content :"))
    i5=float(input("Enter chlorides content :"))
    i6=float(input("Enter free sulfur dioxide content:"))
    i7=float(input("Enter total sulfur dioxide content:"))
    i8=float(input("Enter density :"))
    i9=float(input("Enter pH :"))
    i10=float(input("Enter sulphates content:"))
    i11=float(input("Enter alcohol content:"))
    inp=pd.DataFrame([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11],index=[features]).transpose()
    print("Your input is-")
    print(inp)
    pr=predictions(inp)
    print(f"The predicted wine type is : {pr[0]} and its quality is : {pr[1]} ")
print("Welcome to wine type and quality prediction program!!")
print("_"*55)
while(True):
    ip()
    c=input("Do you want to continue (y/n):")
    if(c=='n' or c=='N'):
        print("Thanks for using our program!")
        break