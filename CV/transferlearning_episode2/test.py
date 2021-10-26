from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import shutil
# Setup the training, validation, and testing dataloaders
from test_loader import create_split_loaders
from setting import work_dir

import torchvision.models as models
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, 10))

num_epochs = 10           # Number of full passes through the dataset
batch_size = 16         # Number of samples in each minibatch
learning_rate = 0.01  
seed = np.random.seed(0) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing


transform = transforms.Compose([
        transforms.Resize([224,224],interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

model = model
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)    

#root_dir = "/home/yipeng/Desktop/movie_tracking/tracking/filter_tracking_result/"
root_dir = "/media/yipeng/Sandisk/movie_tracking/tracking/pipeline_filtered/filter_tracking_result/"
test_loader = create_split_loaders(root_dir,batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)

criterion = nn.CrossEntropyLoss().to(computing_device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def test(val_loader,model,optimizer):
    start = time.time()
    prediction_all = []
    prob_all = []
    fn_all = []
    for mb_count, (val_images, image_fn) in enumerate(val_loader, 0):
        model.eval()
        with torch.no_grad():  
            optimizer.zero_grad()      
            val_images = val_images.to(computing_device)
            outputs = model(val_images)
            output_np = outputs.cpu().detach().numpy()
            prediction  = np.argmax(output_np, axis=1)
            prediction_all.append(prediction)
            fn_all.append(image_fn)
            prob_all.append(output_np)
    prediction_all = np.concatenate(prediction_all)
    prob_all = np.concatenate(prob_all)
    fn_all = np.concatenate(fn_all) 
    return prediction_all, fn_all, prob_all

direct = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/checkpoints/4_model_epoch600.pth"
model_test = model.to(computing_device)
check_point = torch.load(direct)
model_test.load_state_dict(check_point['state_dict'])
prediction, image_fn, prob_all = test(test_loader, model, optimizer)

result_folder = "//media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/resnet_result_episode2/"
for i in range(len(prediction)):
    p = prediction[i]
    fn = image_fn[i]
    prob = prob_all[i][p]
    folder_name = os.path.join(result_folder, str(p))
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    from_fn = os.path.join(root_dir, fn)
    actual_fn = fn.strip().split("/")[-1].split(".")
    actual_fn = actual_fn[0] + "_" + str(prob) + ".jpg"
    to_fn = os.path.join(folder_name, actual_fn)
    shutil.copyfile(from_fn, to_fn)
        

# import matplotlib.pyplot as plt
# from IPython.display import display # to display images
# for minibatch_count, (inputs, labels) in enumerate(train_loader, 0):
#     print(inputs.shape)
#     #inputs = inputs.to(computing_device)
#     #labels = labels.to(computing_device)
#     image = inputs.cpu().detach().numpy()[1].reshape((224,224))*255
    
#     print(image.sum())
#     plt.imshow(image.astype(int),)
#     plt.show()
#     image = inputs.cpu().detach().numpy()[0].reshape((224,224))*255
    
#     print(image.sum())
#     plt.imshow(image.astype(int),)
#     plt.show()
#     image = inputs.cpu().detach().numpy()[2].reshape((224,224))*255
    
#     print(image.sum())
#     plt.imshow(image.astype(int),)
#     plt.show()
    
#     image = inputs.cpu().detach().numpy()[3].reshape((224,224))*255
    
#     print(image.sum())
#     plt.imshow(image.astype(int),)
#     plt.show()
#     #image = Image.fromarray(image.astype(int),"L")
#     #display(image)
#     if minibatch_count >10: 
#         break
    

# >>> loss = nn.CrossEntropyLoss()
# >>> input = torch.randn(3, 5, requires_grad=True)
# >>> target = torch.empty(3, dtype=torch.long).random_(5)
# >>> output = loss(input, target)
# >>> output.backward()


