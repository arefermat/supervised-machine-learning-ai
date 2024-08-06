#Basic for python and fundamentel for Machine learning algorithms
import numpy as np

#Used for random numbers like randomly splitting data set
import random as rn

#More math and open csv files and suck, also works with data frames
import pandas as pd

#This imports a simple LinearRegresion model so i can test scikita vs mine
from scikit-learn import LinearRegression 

#Ploting data and stuff
import seaborn as sns

#Class for the AI bot and all the functions that will train it and stuff
class simple_linr_reg():
    #Initialize the class
    def __init__():
        np.random.seed(9923)
        
    #Class that has functions to prepare the data
    class data_prep():
        #Initalize the class using data
        def __init__(self, data, train=70, val=15, test=15):
            self.data = data
            self.train = train
            self.val = val
            self.test = test
            self.data.columns = data.columns
            self.data.rows = data.rows
            
        #Prep the data by importing it and stuff
        def data_imprt(data=self.data):
            global df
            try:
                df = pd.read_csv(data)
                print("Data : ")
                print(df).to_string()
            except:
                print("DataFrame wasn't able to be created, make sure your data is set up right")
        #Split the data into train, validate, and testing data
        def data_split(train=self.train, val=self.val, test=self.test, x_var, y_var, rows=len(self.data)):
            global x_train, y_train, x_val, y_val, x_test, y_test
            x_train, y_train, x_val, y_val, x_test, y_test = []
            datasets = [x_train, y_train, x_val, y_val, x_test, y_test]
            
            
            for dataset in datasets:
                #test which dataset it is and mulitply that so we know how many times to pick a row
                if dataset == x_train or y_train:
                    set_range = train
                elif dataset == x_val or y_val:
                    set_range = val
                elif dataset == x_test or y_test:
                    set_range = test
                    
                #for loop to loop thorugh the datasets 
                for i in range((rows/100*set_range):
                    #get random numbwr for the row
                    row_nm = rn.randint(1, rows)
                    dataset.append(row_nm)
                
                
                
            
    #Start training the AI with the data and start to validate it 
    class train():
        def __init__():
            pass
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
