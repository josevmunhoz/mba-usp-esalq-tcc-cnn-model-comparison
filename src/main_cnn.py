# Torch imports
import torch
import torch.optim as optim
import torchmetrics
import torchmetrics.classification
from torchvision.models import resnet50, ResNet50_Weights, alexnet, AlexNet_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

# Custom imports
import transformers
from cnn import *
from helper_functions import print_train_time

# LOG Import
from tqdm.auto import tqdm
from timeit import default_timer as timer

# GLOBAL VARIABLES
PATH = "../data/raiox/classified"
TEST_PATH = "../data/raiox/test"
BATCH_SIZE = 32
EPOCHS = 2
SEED = 13
LEARNING_RATE = 0.00004
NUM_CLASS = 2
SIZE = (227, 227) # Usado apenas para o modelo AlexNet construido e não importado
GRAYSCALE = False
MODEL = "AlexNet"

def prepare_data(transform = None):

    ### Create dataset from a specific PATH
    raw_dataset = datasets.ImageFolder(root=PATH,
                                       # If pre-treined trasnform use pre-treined.
                                       transform=transform if transform else transformers.train_transform(size=SIZE, grayscale=GRAYSCALE),
                                       target_transform=None)
    print("\nMetadata of dataset:")
    print(f"Dataset classes: {raw_dataset.classes}")

    # Calculate train and test split
    train_size = int(0.8 * len(raw_dataset))
    valid_size = len(raw_dataset) - train_size
    print(f"Tot imgs to train: {train_size}")
    print(f"Tot imgs to valid: {valid_size}")

    ### Create DataLoader
    print("-----------------------")
    print("\nDataset loaded:")
    torch.manual_seed(seed=SEED)
    train_dataloader, valid_dataloader = random_split(raw_dataset, [train_size, valid_size])

    print("----------------------------------------------------------------------------------")
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    print(train_dataloader.dataset)
    print("\n----------------------------------------------------------------------------------")
    valid_dataloader = DataLoader(dataset=valid_dataloader, batch_size=BATCH_SIZE, shuffle=False)
    print(valid_dataloader.dataset)
    print("\n----------------------------------------------------------------------------------")
    
    print(f"Tot batchs to train: {len(train_dataloader)} | Tot batchs to test: {len(valid_dataloader)}\n")

    return train_dataloader, valid_dataloader


def configureModel(model_name, device):
    ### Model load / creation
    if model_name == "AlexNetConstruida":
        model = AlexNet(num_classes=NUM_CLASS).to(device)
        input_size = (3, 96, 96)
    
    if model_name == "AlexNet":
        model               = torch.hub.load("pytorch/vision", 
                                     "alexnet", 
                                     weights = "DEFAULT").to(device)
        model.classifier[1] = torch.nn.Linear(9216,4096).to(device)
        model.classifier[4] = torch.nn.Linear(4096,1024).to(device)
        model.classifier[6] = torch.nn.Linear(1024,2).to(device)

        transform = AlexNet_Weights.DEFAULT.transforms()
        input_size = (3, 64, 64)

    elif model_name == "Resnet50": # Pre-treined model
        # Using pretrained weights:
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model.fc = nn.Linear(2048, NUM_CLASS)
        model.to(device)
        transform = ResNet50_Weights.DEFAULT.transforms()
        input_size = (3, 64, 64)

    elif model_name == "Efficient_V2_S":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
        model.classifier = nn.Linear(1280, NUM_CLASS).to(device)
        transform = EfficientNet_V2_S_Weights.DEFAULT.transforms()
        input_size = (3, 384, 384)

    return {"model": model, "transform": transform, "input_size": input_size}


def train(optimizer, criterion, model, train_dataloader, valid_dataloader, device):

    total_step = len(train_dataloader)
    train_loss_arr = []
    valid_loss_arr = []

    for epoch in tqdm(range(EPOCHS)):
        for i, (images, labels) in enumerate(train_dataloader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            train_loss_arr.append(loss.item())
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the {} validation images: {} %'.format(len(valid_dataloader), 100 * correct / total)) 


def test(model, transform, device):
    # Get test dataset!
    test_dataset = datasets.ImageFolder(root=TEST_PATH,
                                       # If pre-treined trasnform use pre-treined.
                                       transform=transform if transform else transformers.train_transform(size=SIZE, grayscale=GRAYSCALE))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Tot batchs to Evalute: {len(test_loader)} | Tot images to evaluate: {len(test_dataset)}\n")

    # Set model as evaluation model
    model.eval()
    bacc = torchmetrics.classification.BinaryAccuracy().to(device)
    bcm  = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            bacc(predicted, labels)
            bcm(predicted, labels)

            del images, labels, outputs
        
    print(f"Final Evaluation Metrics:\n Acc: {bacc.compute()} \n Bcm: {bcm.compute()}")
    print(f"plot:")
    fig, ax = bcm.plot()
    print(fig)
    print(ax)


def process():
    start_time = timer()

    # Get the device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print("-----------------------")
    print(f"Device which will process the data: {device}")
    print("-----------------------")

    # Get our model configurations
    output = configureModel(MODEL, device)
    model = output["model"]
    transform = output["transform"]
    input_size = output["input_size"]
    
    ### Prepare and load our data
    train_dataloader, valid_dataloader = prepare_data(transform)

    ### Print out the model params
    print(model.parameters)
    print(summary(model, input_size))

    if True:
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        
        ### Adam optimizer
        optimizer = optim.Adam(model.parameters(),
                            lr=LEARNING_RATE, 
                            weight_decay=0.005)

        # Train
        train(optimizer=optimizer, 
            criterion=criterion, 
            model=model, 
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            device=device)

        # Save the treined model.
        saveModel(MODEL, model)

        ### Calculate training time
        end_time = timer()
        print_train_time(start=start_time, end=end_time, device=device)

        print(f"model loaded: {MODEL}")

        # See how our model performs on unseen data
        test(model, transform, device)


def saveModel(model_name, model):
    # Save the model
    if model_name == "AlexNetConstruida":
        torch.save(model.state_dict(), "AlexnetConstruida.pth")
    if model_name == "AlexNet":
        torch.save(model.state_dict(), "Alexnet.pth")
    elif model_name == "Resnet50":
        torch.save(model.state_dict(), "Resnet.pth")
    elif model_name == "Efficient_V2_S":
        torch.save(model.state_dict(), "Efficient_V2_S.pth")
    else:
        torch.save(model.state_dict(), "default.pth")


if __name__ == "__main__":
    torch.manual_seed(seed=SEED)
    process()

