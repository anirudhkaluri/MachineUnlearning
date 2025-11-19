# ==============================================================
# MACHINE UNLEARNING WITH TIME-DECAY FILTERS (LeNet-256 + MNIST)
# ==============================================================

# --- 1. Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# --- 2. Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Define LeNet-256 architecture ---
class LeNet256(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet256, self).__init__()
        #kernel size is 5X5 #There are 32 different kernels
        #The input is 1 channel which is greyscale 28X28
        #Since there is no padding the image size changes from 24X24 from 28X28
        #The output of conv layer is [batchsize,32,24,24]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        #the previous 32 channels are sent as an input.
        #The input though is changed to 12X12 after max pooling
        #The output of conv layer is [batchsize,64,8,8] as there is no padding.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # Relu is applied on the output of the first convolutional layer
        x = F.max_pool2d(x, 2)    #then we do maxpooling
        x = F.relu(self.conv2(x))        # We then send the maxpooled output to the convolution layer 2 on which relu is again applied
        x = F.max_pool2d(x, 2)           # Second pooling
        x = x.view(x.size(0), -1)        # Flatten for FC layer
        x = F.relu(self.fc1(x))          # Fully connected layer
        x = self.fc2(x)                  # Output layer #not applying ReLu here as CrossEntropyLoss does that internally
        return x

# --- 4. Prepare MNIST dataset (use only 4 classes: 0–3) ---
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Keep only classes 0,1,2,3
train_idx = [i for i, (x, y) in enumerate(mnist_train) if y in [0,1,2,3]]
test_idx = [i for i, (x, y) in enumerate(mnist_test) if y in [0,1,2,3]]

train_data = Subset(mnist_train, train_idx)
test_data = Subset(mnist_test, test_idx)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# --- 5. Train base model on 4 classes ---
model = LeNet256(num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- 6. Baseline accuracy ---
def evaluate(model, dataloader):
    model.eval() #puts the model in evaluation mode
    correct, total = 0, 0
    with torch.no_grad(): #it wont calculate any gradients
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs) #get the outputs
            #for each image you get a tensor of size number of classes
            #dimension of outputs will be [batch size, num classes]

            _, preds = torch.max(outputs, 1)
            # max returns the values and indices of the maximum value of all elements in the output tensor
            #in the above line dim=1 means we are looking for max value along  each row
            #if its 0 we are looking for max value along each column
            correct += (preds == labels).sum().item() #count how many are correct
            total += labels.size(0)
    return 100 * correct / total

print("Training base model...")
try:
    model.load_state_dict(torch.load("lenet256_mnist4.pth"))
    print("Loaded pre-trained model.")
except FileNotFoundError:
    for epoch in range(5):  # small number of epochs for demonstration
        #sets the model to training mode
        #dropout layers become active
        #batch normalization uses batch statistics instead of running statistics.
        #allows gradients to be computed
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs) #basically you input imgs into the model. outputs shape is [batch size, num classes]
            #outputs maintain the computational graph that connects it to model's parameters.
            loss = criterion(outputs, labels) # calculates loss between the outputs and labels. based on cross entropy loss
            loss.backward()  #computes the gradient of the loss w.r.t. model parameters
            #traverses the computational graph backwards
            optimizer.step() #updates model parameters in the direction of the gradient
            total_loss += loss.item() #adds the loss of this batch to the total loss
        print(f"Epoch [{epoch+1}/5], Loss: {total_loss/len(train_loader):.4f}")
        print(f"Base accuracy: {evaluate(model, test_loader):.2f}%")
    torch.save(model.state_dict(), "lenet256_mnist4.pth")



# --- 7. Identify forget and retain datasets ---
forget_class = 0
retain_idx = [i for i, (x, y) in enumerate(train_data) if y != forget_class]
forget_idx = [i for i, (x, y) in enumerate(train_data) if y == forget_class]
retain_data = Subset(train_data, retain_idx)
forget_data = Subset(train_data, forget_idx)

retain_loader = DataLoader(retain_data, batch_size=64, shuffle=True)
forget_loader = DataLoader(forget_data, batch_size=64, shuffle=False)

# --- 8. Compute filter influence for forget class ---
# Target class is the class which we wnat to forget
def compute_filter_influence(model, dataloader, target_class):
    model.eval() #eval disables drop out layers, running statistics instead of batch statistics
    conv1_sum = torch.zeros(32).to(device) #32 filters in conv 1.
    conv2_sum = torch.zeros(64).to(device) #54 filters in conv 2
    count = 0
    #no gradient calculation during the
    with torch.no_grad():
        for imgs, labels in dataloader:
            #in each iteration we get a batch of images and labels from data loader
            imgs, labels = imgs.to(device), labels.to(device) #move to the specified compute device
            #tensor of length 64 for a batch size of 64
            mask = labels == target_class
            if mask.sum() == 0:
                continue
            imgs = imgs[mask] #get only those images which are our forget class
            # Pass through first conv layer
            x = F.relu(model.conv1(imgs)) #x dimension is #number of images, 32,24,24
            #the number of images neednt be 64 because we are taking only the forget class images from the batch
            # Per filter we get the activation.
            # say we have 4 filters. each image will output 4 filters
            # Say filter 1 of image 1 will have 2X2 = 4 individual elements
            # Say there are 5 such images
            # So we add all the individual elements. We get a number. we divide it with 5X4=20
            #Like that we will do for all filters. so the dimensions of conv1_sum will be 4
            #In MNSIT it will be 32
            conv1_sum += x.mean(dim=(0,2,3))
            x = F.max_pool2d(x, 2)
            x = F.relu(model.conv2(x))
            conv2_sum += x.mean(dim=(0,2,3))
            count += 1
    conv1_mean = conv1_sum / count
    conv2_mean = conv2_sum / count
    #conv1_mean and conv2_mean are tensors of size equal to number of filters in respective conv layers
    return {'conv1': conv1_mean, 'conv2': conv2_mean}

#using forget_loader here which has the classes which have to be forgotten
influence = compute_filter_influence(model, forget_loader, forget_class)

# --- 9. Apply time-decay to forget filters ---
def decay_forget_filters(model, influence_dict, lambda_=0.2, topk_ratio=0.25):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Select top-k most activated filters for the forget class
            act = influence_dict[name] #get the activation
            #act will be a tensor of size equal to number of filters in that conv layer
            k = int(len(act) * topk_ratio)
            #Takes act - a tensor of activation values
            #for each filter Uses topk(k) to find the k filters with highest activation values.
            # indices gets just the index positions of those top-k values
            forget_idx = act.topk(k).indices

            # Convert lambda_ to tensor so torch.exp() works
            # If lambda_ = 0.2:
            # 1. -lambda_ = -0.2
            # 2. e^(-0.2) ≈ 0.8187
            decay_factor = torch.exp(torch.tensor(-lambda_, device=module.weight.device))

            with torch.no_grad():
                # Exponential decay applied to those filters
                module.weight.data[forget_idx] *= decay_factor
                if module.bias is not None:
                    module.bias.data[forget_idx] *= decay_factor



# --- 11. Evaluate forgetting effect ---
def class_accuracy(model, dataloader, target_class):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            mask = labels == target_class
            if mask.sum() == 0:
                continue
            imgs, labels = imgs[mask], labels[mask]
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# --- 10. Simulate unlearning with time-based decay ---
#epochs_unlearn = 5

#remove the forget class from test data to evaluate retain classes
retain_test_idx = [i for i, (x, y) in enumerate(test_data) if y != forget_class]
retain_test_data = Subset(test_data, retain_test_idx)
retain_test_loader = DataLoader(retain_test_data, batch_size=256, shuffle=False)
epochs_to_unlearn= [1,3,5,7,10]
retain_acc=[]
forget_acc=[]

for i in range(len(epochs_to_unlearn)):
    epochs_unlearn = epochs_to_unlearn[i]
    lambda_base = 0.2
    # Reset model to pre-unlearning state
    model.load_state_dict(torch.load("lenet256_mnist4.pth"))
    print(f"\nUnlearning for {epochs_unlearn} epochs with time-decay...")
    for epoch in range(1, epochs_unlearn + 1):
        lambda_t = lambda_base * epoch           # decay increases over time
        decay_forget_filters(model, influence, lambda_=lambda_t)
        print(f"Applied decay step {epoch} with λ={lambda_t:.2f}")

        # Fine-tune slightly on retain dataset
        model.train()
        for imgs, labels in retain_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    acc_retain = evaluate(model, retain_test_loader)
    acc_forget = class_accuracy(model, test_loader, forget_class)
    print(f"After time-decay unlearning with {epochs_unlearn} Epochs:")
    print(f"  Accuracy (retain classes): {acc_retain:.2f}%")
    print(f"  Accuracy (forgotten class {forget_class}): {acc_forget:.2f}%")
    retain_acc.append(acc_retain)
    forget_acc.append(acc_forget)


# --- 12. Optional: visualize decay effect ---

plt.figure(figsize=(10, 6))
plt.plot(epochs_to_unlearn, retain_acc, marker='o', label='Retain Classes (0,1,2,3)', linewidth=2, color='skyblue')
plt.plot(epochs_to_unlearn, forget_acc, marker='s', label=f'Forget Class ({forget_class})', linewidth=2, color='lightcoral')
plt.xlabel('Number of Unlearning Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Effect of Unlearning Epochs on Model Performance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

