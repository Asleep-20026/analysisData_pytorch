from data_processing import *
from torch.utils.data import random_split, DataLoader, TensorDataset
from perceptron import *
import torch
import torch.nn.functional as F 

class Trainner:
    def __init__(self, data_processor, device, num_epochs=10):
        self.data_processor = data_processor
        self.device = device
        self.num_epochs = num_epochs

        # Split to Train, Validate and Test sets using random_split
        self.train_batch_size = 10
        self.number_rows = len(self.data_processor.input)
        self.test_split = int(self.number_rows * 0.3)
        self.validate_split = int(self.number_rows * 0.2)
        self.train_split = self.number_rows - self.test_split - self.validate_split
        self.train_set, self.validate_set, self.test_set = random_split(
            self.data_processor.data, [self.train_split, self.validate_split, self.test_split])

        # Create DataLoader to read the data within batch sizes and put into memory.
        self.train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True)
        self.validate_loader = DataLoader(self.validate_set, batch_size=1)
        self.test_loader = DataLoader(self.test_set, batch_size=1)

        # Define model parameters
        self.input_size = list(self.data_processor.input.shape)[1]
        self.output_size = len(self.data_processor.labels)

        # Instantiate the model 
        self.model = Network(self.input_size, self.output_size)

        # Define your execution device 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        print("The model will be running on", self.device, "device\n") 

        # Move the model to the selected device
        self.model.to(self.device)

        # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

    # Function to save the model 
    def save_model(self): 
        path = "./NetModel.pth" 
        torch.save(self.model.state_dict(), path)
        
    # Training Function 
    def train(self,num_epoch): 
        best_accuracy = 0.0 

        print("Begin training...") 
        for epoch in range(1, self.num_epochs + 1): 
            running_train_loss = 0.0 
            running_accuracy = 0.0 
            running_val_loss = 0.0 
            total = 0 

            # Training Loop 
            for data in self.train_loader: 
                inputs, outputs = data
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                self.optimizer.zero_grad()   
                predicted_outputs = self.model(inputs)   
                train_loss = self.loss_fn(predicted_outputs, outputs)   
                train_loss.backward()   
                self.optimizer.step()        
                running_train_loss += train_loss.item()  

            # Calculate training loss value 
            train_loss_value = running_train_loss / len(self.train_loader) 

            # Validation Loop 
            with torch.no_grad(): 
                self.model.eval() 
                for data in self.validate_loader: 
                    inputs, outputs = data 
                    inputs = inputs.to(self.device)
                    outputs = outputs.to(self.device)
                    predicted_outputs = self.model(inputs) 
                    val_loss = self.loss_fn(predicted_outputs, outputs) 

                    _, predicted = torch.max(predicted_outputs, 1) 
                    running_val_loss += val_loss.item()  
                    total += outputs.size(0) 
                    running_accuracy += (predicted == outputs).sum().item() 

            # Calculate validation loss value 
            val_loss_value = running_val_loss / len(self.validate_loader) 

            # Calculate accuracy
            accuracy = (100 * running_accuracy / total)     

            # Save the model if the accuracy is the best 
            if accuracy > best_accuracy: 
                self.save_model() 
                best_accuracy = accuracy 

            # Print the statistics of the epoch 
            print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value, 'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is %d %%' % accuracy)
