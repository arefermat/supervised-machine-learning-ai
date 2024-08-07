o#Basic for python and fundamentel for Machine learning algorithms
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
class simple_linr_regr():
    #Initialize the class
    def __init__():
        np.random.seed(9923)
        
    #Class that has functions to prepare the data
    class data_prep():
        #Initalize the class using data
        def __init__(self, data, train=70, test=15):
            self.data = data
            self.train = train
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
        def data_split(train=self.train, test=self.test, x_var, y_var, rows=len(self.data), data=self.data):
            global x_train, y_train, x_test, y_test
            x_train = np.array()
            y_train = np.array()
            x_test = np.array()
            y_test = np.array()
            xtest = ["test"]
            ytest = ["test"]
            xtrain = ["train"]
            ytrain = ["train"]
            datasets = [xtrain, ytrain, xtest, ytest]
            for dataset in datasets:
                #test which dataset it is and mulitply that so we know how many times to pick a row
                if dataset == "train":
                    set_range = train
                elif dataset == "test":
                    set_range = test
                dataset = []
                #for loop to loop thorugh the datasets 
                for i in range((rows/100*set_range)):
                    #get random number for the row
                    row_nm = rn.randint(1, rows)
                    for datas in dataset:
                        if row_nm == datas:
                            pass
                        else:
                            dataset.append(row_nm)
                    
            for dataset in datasets:
                if dataset == xtest:
                    for datas in dataset:
                        a = np.append(x_test, [data.values[datas][x_var]])
                        x_test = a
                    x_test.reshape(-1, 1)
                elif dataset == ytest:
                    for datas in dataset:
                        a = np.append(y_test, [data.values[datas][y_var]])
                        y_test = a
                    y_test.reshape(-1, 1)
                elif dataset == xtrain:
                    for datas in dataset:
                        a = np.append(x_train, [data.values[datas][x_var]])
                        x_train = a
                    x_train.reshape(-1, 1)
                elif dataset == ytrain:
                    for datas in dataset:
                        a = np.append(y_train, [data.values[datas][y_var]])
                        y_train = a
                    y_train.reshape(-1, 1)
                    
                
                
                
                
            
    #Start training the AI with the data and start to validate it 
    class train():
        data = data.dropna()
        
        def __init__():
            self.parameters = {}
            self.loss = []
            
        def cost_func(self, predictions, y_train):
            #Get the MSE of the output
            cost = np.mean((y_train - predictions) ** 2) 
            return cost 
                
        def fwd_propg(self, x_train):
            m = self.parameters['m'] 
            c = self.parameters['c'] 
                
            predictions = np.multiply(m, x_train) + c 
            return predictions
                
        def back_propg(self, x_train, y_train, predictions):
            derivatives = {}
            df = (predictions - y_train)
                
            dm = 2 * np.mean(np.multiply(x_train, df))
                
            dc = 2 * np.mean(df)
                
            derivatives["dm"] = dm
            derivatives["dc"] = dc
                
            return derivatives
                
                
        #Test the ai with the testing data set 
        def update_perams(self, derivatives, learning_rate):
            self.parameters["m"] = self.parameters["m"] - learning_rate * derivatives["dm"]
            self.parameters["c"] = self.parameters["c"] - learning_rate * derivatives["dc"]
            
        #Final function so that you can give it data and some other things to do everything in one function, preditcs what will happen
        def train(self, x_train, y_train, learning_rate, iters):
            
            self.parameters["m"] = np.random.uniform(0, 1) * -1
            self.parameters["c"] = np.random.uniform(0, 1) * -1
            
            for i in range(iters):
                predictions = self.fwd_propg(x_train)
                
                cost = self.cost_func(predictions, y_train)
                
                derivatives = self.back_propg(x_train, y_train, predictions)
                
                self.update_perams(derivatives, learning_rate)
                
                self.loss.append(cost)
                print(f"Iteration = {i + 1}, Loss = {cost}")
                
                return self.parameters, self.loss
                
                
if __name__ == "__main__":
    
    model = simple_linr_regr()
    lern_rt = int(input("Learning rate : "))
    itrs = int(input("Iterations : "))
    parameters, loss = model.train(x_train, y_train, learning_rate=lern_rt, iters=itrs)
    
    print("Trained parameters :", paramters)
    print("Final loss :", loss[-1])
                
                
