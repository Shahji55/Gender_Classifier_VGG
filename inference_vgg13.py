import torch
from torchvision import transforms
import cv2
from PIL import Image
from torchvision import transforms
import time

import torchvision.models as models

from model_vgg13 import VGGnet
import os
import torch.nn as nn

import bleedfacedetector as fd

def loadModel(weights):

    model = VGGnet(in_channels=3, num_classes=2)
    # model.load_state_dict(state, torch.load(weights))
    # model.load_state_dict(torch.load(weights))

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(model['optimizer_state_dict'])
    #epoch = model['epoch']
    #loss = model['loss']
    #arch = model['arch']
    model.cuda()
    model.eval()
    return model

def inferenceGenderModel(img , model):

    trans = transforms.Compose([
            transforms.Resize((180,180)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # cv2 to PIL conversion
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    out = model(img.cuda())
    # out = model(img)
    print("out:  ", out)
    _, index = torch.max(out, 1)
    return index.item()

def image_inference(model):
    image = cv2.imread('test.jpg')
    t1=time.time()
    output = inferenceGenderModel(image, model)
    t2=time.time()
    print(t2-t1)
    mapper=['female', 'male']
    output= mapper[output]
    cv2.putText(image, output, (50, 50), 2, 1, (0, 0, 255))

    cv2.imwrite('output.jpg', image)
    #cv2.imshow('hello', image)
    #cv2.waitKey(1)

def images_dir_inference(model):

    images_path = os.getcwd() + '/data/Gender/val/male/'
    write_path = os.getcwd() + '/data/inference_results/gender_val_set/male/'

    mapper=['female', 'male']
    male_counter = 0
    female_counter = 0

    list = os.listdir(images_path) # dir is your directory path
    number_files = len(list)

    for img in os.listdir(images_path):
        img_path = images_path + img
        img_name = img.split('.')[0]

        print(img_path)
        image = cv2.imread(img_path)

        t1=time.time()
        output = inferenceGenderModel(image, model)
        t2=time.time()
        print(t2-t1)
        
        output= mapper[output]
        print(output)

        if output == 'male':
            male_counter = male_counter + 1
        elif output == 'female':
            female_counter = female_counter + 1

        cv2.putText(image, output, (50, 50), 2, 1, (0, 0, 255))

        output_img = write_path + img_name + '.jpg'
        cv2.imwrite(output_img, image)

    print("Total images: ", number_files)
    print("Male predictions: ", male_counter)
    print("Female predictions: ", female_counter)

if __name__ == "__main__":
    model = loadModel('./58-epochs-gender-vgg13-orig-model/model_best.pth.tar')

    images_dir_inference(model)


         