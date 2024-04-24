import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def argument_parser():
    #using argument parsing to recieve parameters like image and checkpoint file from the user #through command line
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)#the default for the epochs is 5 , if user want to change it is allowed just write
    #python train.py --epochs any_number_user_wants
    parser.add_argument('--save_dir', dest="save_dir", action="store")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)#if not specified by user then it will be 120
    
    
    args = parser.parse_args()
    return args



def training_transformations(train_dir):
    #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html#imagefolder 
# from the Imagenet example provided by pytorch documentation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html#imagefolder
     #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data



def test_transformations(test_dir):
    #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html#imagefolder 
# from the Imagenet example provided by pytorch documentation
    test_transforms = transforms.Compose([transforms.Resize(size=250),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean = [ 0.485,                                                             0.456, 0.406 ],
                                                          std = [ 0.229, 0.224, 0.225 ]) ])
    #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html#imagefolder
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    

def data_loader(data, train=True):
    #resource:https://pytorch.org/docs/0.3.0/torchvision/datasets.html
    if train: 
        #specifying the batch_size as 16 ,and shuffle the images
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=16)
    return loader





def checking_gpu(gpu_arg):
    #checking if user has given any argument for gpu_arg , if not given then device used will be cpu
    if not gpu_arg:
        return torch.device("cpu")
    #checking if cuda is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0" )
    else:
         device = torch.device("cpu")

    return device


def primaryloader_model(architecture="vgg16"):
    
    #uses pretrained model vgg16
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
     #freezes gradients   
    for param in model.parameters():
        param.requires_grad = False 
    return model


def classifier( hidden_units,architecture="vgg16"):
    #uses pretrained model vgg16
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
     #freezes gradients   
    for param in model.parameters():
        param.requires_grad = False 
    

    classifier = nn.Sequential(
             nn.Linear(25088, 120), #hidden layer 1 sets output to 120
             nn.ReLU(),           #relu activation function for the first hidden layer
            nn.Dropout(0.5),       #for regularization
            nn.Linear(120, 90), #hidden layer 2 output to 90
            nn.ReLU(),
            nn.Linear(90,70), #hidden layer 3 output to 70
            nn.ReLU(),
            nn.Linear(70,102),#output size = 102
           nn.LogSoftmax(dim=1))# For using NLLLoss()

    model.classifier = classifier
    return classifier,model



def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for i, (inputs, labels) in enumerate(testloader):
        
        inputs=inputs.to(device)
        labels = labels.to(device)
        
        output = model.forward(inputs)
        #used to calculate and accumulate the test loss during the evaluation
        #loss: refers to a loss function or criterion used to compute the loss 
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        #computesthe maximum probability by ps.max(dim=1)
        #used [1] at the end of the line to extract the second element in the tuple
        #it then compare it to the true labels
        equality = (labels.data == ps.max(dim=1)[1])
        #converts the boolean values in the equality tensor into floating-point values (1.0 for              True and 0.0 for False) by using .type(torch.FloatTensor)
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy




def network_trainer(Model, Trainloader, Testloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):
    #if not specified by the user then it will be 18
    #if type(Epochs) == type(None):
       # Epochs = 18
       # print("Number of Epochs specificed as 18.")    
 
    print("Training process starting ")

    # Train the Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() 
        
        for i, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward passes
            outputs = Model.forward(inputs)
            #loss function 
            loss =Criterion(outputs, labels)
            #backward pass
            loss.backward()
            #optimizer
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    #validating the model performance with test_loader
                    valid_loss, accuracy = validation(Model, Testloader, Criterion,Device)
                #printing out the values of Epochs, training loss, validation_loss, accuracy
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/Print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(Testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(Testloader)))
            
                running_loss = 0
                Model.train()

    return Model



#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct=0
    total = 0
    #freezing gradients
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = Model(images)
            #outputs.data ==>(batch_size, num_classes)
        #this line calculates the maximum value along dimension 1 in outputs.data
        #"_" represents the maximum values along dimension 1, which are not being stored in a variable.
            _, predicted = torch.max(outputs.data, 1)
             #labels.size(0)represents number of examples in the current batch
            total += labels.size(0)
            #This line increments the number of correctly classified images as when predicted==labels
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))
    

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, can't save the model.")
    else:
        if isdir(Save_Dir):
            #mapping of classes to indices 
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')
            
      

        else: 
            print("Directory not found, model will not be saved.")

def main():
     
    
    args = argument_parser()
    
    # Set directory for train data,valid data and test data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # transformations on train_data, test_data_valid_data
    train_data = training_transformations(train_dir)
    valid_data = test_transformations(valid_dir)
    test_data = test_transformations(test_dir)
    #dataloaders for every dataset
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    #getting the architecture for the model
    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    #checking gpu
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    #loss function
    criterion = nn.NLLLoss()
    #specifying optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    #train the model then validate the model on test_images and then saving the checkpoint for inference later
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process completed")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data)
if __name__ == '__main__': main()