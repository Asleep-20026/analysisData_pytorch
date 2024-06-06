from data_processing import *
from perceptron import *
from train import *


class Testing:
    def test(): 
    # Load the model that we saved at the end of the training loop 
        model = Network(Network.input_size, Network.output_size) 
        path = "NetModel.pth" 
        model.load_state_dict(torch.load(path, map_location=Trainner.device))
        model.to(Trainner.device)  # Asegúrate de que el modelo esté en el dispositivo correcto
        
        running_accuracy = 0 
        total = 0 
    
        with torch.no_grad(): 
            for data in Trainner.test_loader: 
                inputs, outputs = data 
                
                # Mueve los datos de entrada y salida al dispositivo
                inputs = inputs.to(Trainner.device)
                outputs = outputs.to(Trainner.device)
                
                # Verifica que el modelo y los datos estén en el mismo dispositivo
                assert next(model.parameters()).device == inputs.device
                
                predicted_outputs = model(inputs) 
                _, predicted = torch.max(predicted_outputs, 1) 
                total += outputs.size(0) 
                running_accuracy += (predicted == outputs).sum().item() 
    
        print('Accuracy of the model based on the test set of', Trainner.test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))
        
    def test_species():
        # Carga el modelo que guardaste al final del bucle de entrenamiento
        Trainner.model = Network(Network.input_size, Network.output_size)
        path = "NetModel.pth"
        Trainner.model.load_state_dict(torch.load(path, map_location=Trainner.device))
        Trainner.model.to(Trainner.device)

        labels_length = len(DataProceesor.labels)
        labels_correct = list(0. for i in range(labels_length))
        labels_total = list(0. for i in range(labels_length))

        with torch.no_grad():
            for data in Trainner.test_loader:
                inputs, outputs = data
                inputs = inputs.to(Trainner.device)  # Mueve las entradas al mismo dispositivo que el modelo
                outputs = outputs.to(Trainner.device)  # Mueve las salidas al mismo dispositivo que el modelo
                predicted_outputs = Trainner.model(inputs)
                _, predicted = torch.max(predicted_outputs, 1)

                label_correct_running = (predicted == outputs).squeeze()
                label = outputs[0]
                if label_correct_running.item():
                    labels_correct[label] += 1
                labels_total[label] += 1

        # Resto del código
    
        label_list = list(DataProceesor.labels.keys()) 
        for i in range(Network.output_size): 
            print('Accuracy to predict %5s : %2d %%' % (label_list[i], 100 * labels_correct[i] / labels_total[i]))