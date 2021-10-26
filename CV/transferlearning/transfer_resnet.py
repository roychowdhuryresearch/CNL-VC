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
# Setup the training, validation, and testing dataloaders
from data_loader import create_split_loaders
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
        #transforms.RandomResizedCrop(224),
        #transforms.ToPILImage('L'),
        transforms.Resize([224,224],interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(), 
        transforms.RandomRotation(30),
        transforms.RandomCrop((170, 170)),
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

root_dir = "/home/yipeng/Desktop/movie_tracking/tracking/group_result_network/"
train_loader, val_loader, test_loader = create_split_loaders(root_dir,batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)

criterion = nn.CrossEntropyLoss().to(computing_device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def validate(val_loader,model,optimizer):
    start = time.time()
    sum_loss = 0.0
    list_sum_loss = []
    minibatch_acc = 0.0
    num = 0
    for mb_count, (val_images, val_labels) in enumerate(val_loader, 0):
        model.eval()
        with torch.no_grad():  
            optimizer.zero_grad()      
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
            val_labels = val_labels.long()
            outputs = model(val_images)
            loss = criterion(outputs,val_labels)
            output_np = outputs.cpu().detach().numpy()
            label_np = val_labels.cpu().detach().numpy()
            minibatch_acc += accuracy(output_np,label_np)
            sum_loss += loss
    print("validation time = ", time.time()-start)
    print("vali_accuracy = ", 1.0*minibatch_acc/mb_count)    
    return 1.0*sum_loss/mb_count  

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    #print(outputs, labels)
    return np.sum(outputs==labels)/float(labels.size)


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    total_loss = []
    avg_minibatch_loss = []
    total_vali_loss = []
    tolerence = 3
    i = 0 
    for epoch in range(num_epochs):
        N = 100
        M = 100
        N_minibatch_loss = 0.0    
        best_loss = 100
        early_stop = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            scheduler.step()
            # Iterate over data.
            for minibatch_count, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.to(computing_device)
                labels = labels.to(computing_device)
                labels = labels.long()
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.long()
                    loss = criterion(outputs,labels)
                    #loss = criterion(torch.max(outputs,1)[1],torch.max(labels, 1)[1])
                    N_minibatch_loss += loss
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                
                # statistics
                # training stats
                if minibatch_count % N == 0 and minibatch_count!=0:    

                    # Print the loss averaged over the last N mini-batches    
                    N_minibatch_loss /= N
                    print('Epoch %d, average minibatch %d loss: %.3f' %
                        (epoch + 1, minibatch_count, N_minibatch_loss))

                    # Add the averaged loss over N minibatches and reset the counter
                    avg_minibatch_loss.append(N_minibatch_loss)
                    
                    avg_minibatch_loss_1 = np.array(avg_minibatch_loss)
                    np.save('avg_minibatch_loss_new', avg_minibatch_loss_1)
                    
                    N_minibatch_loss = 0.0

                    output_np = outputs.cpu().detach().numpy()
                    label_np = labels.cpu().detach().numpy()

                    accuracy_train = accuracy(output_np, label_np)
                    print('accuracy',accuracy_train)
                    #print('accuracy, precision, recall', accuracy, precision, recall)
                
                #Validation
                if minibatch_count % M == 0 and minibatch_count!=0: 
                    #model = torch.load('./checkpoint')
                    save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                },
                                filename=os.path.join(work_dir, 'checkpoints','%d_model_epoch%d.pth' % (epoch , minibatch_count)))
                    v_loss = validate(val_loader,model,optimizer).item()
                    total_vali_loss.append(v_loss)

                    total_vali_loss_1 = np.array(total_vali_loss)
                    np.save(os.path.join(work_dir, 'total_vali_loss'), total_vali_loss_1)                    

                    if total_vali_loss[i] > best_loss and i != 0:
                        early_stop += 1
                        if early_stop == tolerence:
                            print('early stop here')
                            break
                    else:
                        best_loss = total_vali_loss[i] 
                        early_stop = 0
                    i = i + 1
            print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")
    
    avg_minibatch_loss = np.array(avg_minibatch_loss)
    np.save('avg_minibatch_loss_new', avg_minibatch_loss)

    total_vali_loss = np.array(total_vali_loss)
    np.save('total_vali_loss_new', total_vali_loss)  
    print("total_vali_loss")
    print(total_vali_loss)
    print("avg_minibatch_loss")
    print(avg_minibatch_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s '.format(
        time_elapsed // 60, time_elapsed % 60))

def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)


train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)


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


