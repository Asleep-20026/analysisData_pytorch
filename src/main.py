from data_processing import *
from perceptron import *
from train import *

if __name__ == "__main__": 
    num_epochs = 10
    data_processor = DataProceesor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainner = Trainner(data_processor, device, num_epochs)
    trainner.train() 
    print('Finished Training\n') 
    testing = trainner.Testing()
    testing.test() 
    testing.test_species()
