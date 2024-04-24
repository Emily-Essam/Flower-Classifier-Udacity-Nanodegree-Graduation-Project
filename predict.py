import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from train import test_transformations
from torchvision import models
from torchvision import datasets, transforms

def argument_parser():
    #using argument parsing to recieve parameters like image and checkpoint file from the user #through command line
    parser = argparse.ArgumentParser(description="predict.py")
    
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)#has to be given by user
    parser.add_argument('--checkpoint',type=str,help='checkpoint file as str.',required=True)#has to be given by user
    #if user didn't give the top k elements number , then the default will be 5
    parser.add_argument('--top_k',type=int,help='Choose top K  as int.',default=5)
    #if user does't use his own labels we will use "cat_to_name.json"as our default
    parser.add_argument('--flower_category_names', dest="category_names", action="store", default='cat_to_name.json')
    #for gpu needed for training the model
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args


def loading_checkpoint(checkpoint_path):
  #this function is for loading the checkpoint saved when training the model to use it for inference  
     checkpoint = torch.load("my_checkpoint.pth")
    
     model = models.vgg16(pretrained=True)
     model.name = "vgg16"  #model name used id vgg16
    
    
    #we don't need gradients for inference
     for param in model.parameters(): 
        param.requires_grad = False
    
    # Loading the parameters from checkpoint
     model.class_to_idx = checkpoint['class_to_idx']
     model.load_state_dict(checkpoint['state_dict'])
     model.classifier = checkpoint['classifier']
     
    
     return model

def process_image(image_path):
    ''' given a PIL image ,this function does the required transformations before giving the           image to the pytorch model,
        returns a Numpy array
    '''
    transformations=transforms.Compose([transforms.Resize(size=250),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                           std = [ 0.229, 0.224, 0.225 ]) ])
    test_image = PIL.Image.open(image_path)
    with test_image as image:  
        image = transformations(image).numpy()
        
    return image

def predict(image_tensor, model,device,cat_to_name ,top_k):
    ''' Predict the class (or classes) of an image .
    
    image_tensor: string. Path to image.
    model: pytorch neural network.
    device:gpu,cpu
    cat_to_name:the labels json file
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities, top_labels, top_flowers
    '''
    
    # we don't need Gpu for inferencing
    model.to("cpu")
    
    
    model.eval();

    #np.expand_dims(processed_image, axis=0)==> Deep learning models often expect input data in      batches, so expand_dims method 
    #adds an extra dimension to the image to make it compatible with the model's input shape
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_tensor), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    #used to perform forward pass, model is passed through the whole network to produce output 
    log_probability = model.forward(torch_image)

    #the log probabilities achieved from the previous step we compute the exponential to convert it into positive value
    linear_probability = torch.exp(log_probability)

     #top_probability==>This tensor contains the top-K probabilities in descending order
    #top_labels==>contains class labels corressponds to the top_k probilities
    top_probability, top_labels = linear_probability.topk(top_k)
    
     #deattaches the top_probability from the gradients then converts it into numpy array
    top_probability = np.array(top_probability.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    class_to_idx = model.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    top_labels=[]
    top_flowers=[]
    for label in top_labels:
        top_labels.append(idx_to_class[label]) 
    for label in top_labels:
        top_flowers.append(cat_to_name[label])
    
    return top_probability, top_labels, top_flowers


def print_probability(flowers, probs):
     #converts two lists, probs and flowers, into a dictionary-like format 
    for i, j in enumerate(zip(probs, flowers)):
        print ("order {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], j[0]*100) # prints the flower name (from j[1]) and its likelihood (from j[0]) as a percentage
        #we can use ceil method to round the liklihoodup to the nearest integer


def main():
    
    args = argument_parser()
    #loading the json file and opening it as read only
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
    #loading the checkpoint by using the path of it given by user in command line
    model = loading_checkpoint(args.checkpoint)
    #processing the image using the process_image using the image path given by the user in          command line app
    image_tensor = process_image(args.image)
    #check the gpu
    device = check_gpu(gpu_arg=args.gpu);
    #predecting and returning the top_probability, top_labels, top_flowers
    top_probability, top_labels, top_flowers = predict(args.image, model,device, cat_to_name,args.top_k)
    
    #printing the probability
    print_probability(top_flowers, top_probability)

if __name__ == '__main__': main()