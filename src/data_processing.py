import os
import pandas as pd

class DataProceesor:

    def __init__(self):
        # Loading the Data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_dir = os.path.join(script_dir, '..', 'data', 'Iris_dataset.xlsx')

        self.df = pd.read_excel(excel_dir)
        print('Take a look at sample from the dataset:')
        print(self.df.head())

        # Let's verify if our data is balanced and what types of species we have
        print('\nOur dataset is balanced and has the following values to predict:')
        print(self.df['Iris_Type'].value_counts())

        # Convert Iris species into numeric types: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2.
        self.labels = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} 
        self.df['IrisType_num'] = self.df['Iris_Type']  # Create a new column "IrisType_num"
        self.df.IrisType_num = [self.labels[item] for item in self.df.IrisType_num]  # Convert the values to numeric ones

        # Define input and output datasets
        self.input = self.df.iloc[:, 1:-2]  # We drop the first column and the two last ones.
        print('\nInput values are:')
        print(self.input.head())
        self.output = self.df.loc[:, 'IrisType_num']  # Output Y is the last column
        print('\nThe output value is:')
        print(self.output.head())