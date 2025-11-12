import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


#connect images to number and normalized it
daddy_1 = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


#load data
daddy_2 = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=daddy_1)        #train
daddy_3 = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=daddy_1)        #test
daddy_2_l = torch.utils.data.DataLoader(daddy_2, batch_size=64, shuffle=True)
daddy_3_l = torch.utils.data.DataLoader(daddy_3, batch_size=64, shuffle=True)


#shape of train & label
daddy_iter = iter(daddy_2_l)
daddy_im, daddy_ls = next(daddy_iter)

print(daddy_im.shape)
print(daddy_ls.shape)


#display
plt.imshow(daddy_im[0].numpy().squeeze(), cmap='gray_r');

daddy_fig = plt.figure()     #figure
daddy_nuts = 60     #nums of images
for index in range(1, daddy_nuts + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(daddy_im[index].numpy().squeeze(), cmap='gray_r')


#Build Neural Network
daddy_in = 784      #inputs
daddy_hid = [128, 64]       #hidden layers
daddy_out = 10      #outputs

daddy_here = nn.Sequential(nn.Linear(daddy_in, daddy_hid[0]),       #classification model
                      nn.ReLU(),        #ReLU activation
                      nn.Linear(daddy_hid[0], daddy_hid[1]),        #linear layer 1
                      nn.ReLU(),        #ReLU activation
                      nn.Linear(daddy_hid[1], daddy_out),       #linear layer 2
                      nn.LogSoftmax(dim=1))     #linear layer 3 with LogSoftmax activation
print(daddy_here)


#define the negative log-likelihood loss
daddy_crit = nn.NLLLoss()
daddy_im, daddy_ls = next(iter(daddy_2_l))
daddy_im = daddy_im.view(daddy_im.shape[0], -1)

daddy_lps = daddy_here(daddy_im) #log probabilities
daddy_loss = daddy_crit(daddy_lps, daddy_ls) #calculate the NLL loss


#adjusting weights to minimize the loss
print('Before backward pass: \n', daddy_here[0].weight.grad)
daddy_loss.backward()
print('After backward pass: \n', daddy_here[0].weight.grad)


#training process
daddy_opt = optim.SGD(daddy_here.parameters(), lr=0.003, momentum=0.9)      #optimizer
daddy_4 = time()      #initial time
epochs = 15
for e in range(epochs):
    running_loss = 0
    for daddy_im, daddy_ls in daddy_2_l:
        # Flatten MNIST images into a 784 long vector
        daddy_im = daddy_im.view(daddy_im.shape[0], -1)
    
        # Training pass
        daddy_opt.zero_grad()
        
        daddy_out = daddy_here(daddy_im)
        daddy_loss = daddy_crit(daddy_out, daddy_ls)
        
        #This is where the model learns by backpropagating
        daddy_loss.backward()
        
        #And optimizes its weights here
        daddy_opt.step()
        
        running_loss += daddy_loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(daddy_2_l)))
print("\nTraining Time (in minutes) =",(time()-daddy_4)/60)


#show image and class prob
def daddy_view(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


#testing & evaluation
daddy_im, daddy_ls = next(iter(daddy_3_l))

img = daddy_im[0].view(1, 784)
with torch.no_grad():
    daddy_lps = daddy_here(img)

ps = torch.exp(daddy_lps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
daddy_view(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for daddy_im,daddy_ls in daddy_3_l:
  for i in range(len(daddy_ls)):
    img = daddy_im[i].view(1, 784)
    with torch.no_grad():
        daddy_lps = daddy_here(img)

    
    ps = torch.exp(daddy_lps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = daddy_ls.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


#save model
torch.save(daddy_here, './my_mnist_model.pt') 