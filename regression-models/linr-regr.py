#Basic for python and fundamentel for Machine learning algorithms
import numpy as np

#More math and open csv files and suck, also works with data frames
import pandas as pd

#This imports a simple LinearRegresion model so i can test scikita vs mine
from scikit-learn import LinearRegression 

#Ploting data and stuff
import seaborn as sns

#Class for the AI bot and all the functions that will train it and stuff
class linr_reg():
    #Initialize the class
    def __init__():
        pass
        
    #Class that has functions to prepare the data
    class data_prep():
        #Initalize the class using data
        def __init__(self, data, train=0.7, val=0.15, test=0.15):
            self.data = data
            self.train = train
            self.val = val
            self.test = test
            
        #Prep the data by importing it and stuff
        def data_imprt(data=self.data):
            df = pd.DataFrame(data, columns=data.columns)
            print("Data : ")
            print(df).to_string()
        #Split the data into train, validate, and testing data
        def data_split(train=self.train, val=self.val, test=self.test)):
            pass
            
    #Start training the AI with the data and start to validate it 
    class train():
        #Train the AI using the training dataset
        def train_mlai():
            pass
        #Validate the AI and tweak andything if its getting stuff wrong
        def validate_data():
            pass
        #Test the ai with the testing data set 
        def test_data():
            pass
            
    #Final function so that you can give it data and some other things to do everything in one function, preditcs what will happen
    def predict():
        pass
