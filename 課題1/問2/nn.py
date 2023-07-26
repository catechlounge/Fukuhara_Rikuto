import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import normal_distribution
from sklearn.metrics import mean_absolute_error
import numpy as np

'''
学習している中の最高精度
Epoch [18/1000], Loss: 1.4104, val_loss: 1.2191, val_acc: 0.6429
xは、normal_distribution.pyで心拍指標を正規分布に従わせて正規化している。
'''

class HeartRate_Dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class DNN(nn.Module):
    def __init__(self,input_size):
        super(DNN, self).__init__()

        self.layer1 = nn.Linear(input_size, 256)
        nn.init.kaiming_normal_(self.layer1.weight)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        nn.init.kaiming_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.layer3.weight)
        self.layer4 = nn.Linear(64, 32)
        nn.init.kaiming_normal_(self.layer4.weight)
        self.layer5 = nn.Linear(32, 7)  # Output size of 7
        nn.init.kaiming_normal_(self.layer5.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        x = torch.relu(x)
        x = self.layer5(x)

        return x

csv_path = '../covtype.csv'
heartrate_data = pd.read_csv(csv_path)

base_colums = heartrate_data[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40"]]
x = normal_distribution.preprocess(csv_path=csv_path,base_columns=base_colums,visualize=False).values
y = heartrate_data["Cover_Type"].values-1

train_X,test_X,train_Y,test_Y = train_test_split(x,y,random_state=0)

train_dataset = HeartRate_Dataset(train_X,train_Y)
test_dataset = HeartRate_Dataset(test_X,test_Y)

train_dataloader  = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = train_X.shape[1]
model = DNN(input_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0025,weight_decay=0.0025)

num_epochs = 1000
best_accuracy = 0
cur_accuracy = 0

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
mae_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    val_labels = []
    val_predictions = []

    model.train()
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_dataloader.dataset)
    avg_train_acc = train_acc / len(train_dataloader.dataset)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()

            # Add the labels and predictions to the lists
            val_labels.append(labels.cpu().numpy())
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            val_predictions.append(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(test_dataloader.dataset)
    avg_val_acc = val_acc / len(test_dataloader.dataset)

    # Calculate the mean absolute error
    val_labels = np.concatenate(val_labels)
    val_predictions = np.concatenate(val_predictions)
    mae = mean_absolute_error(val_labels, val_predictions)
    mae_list.append(mae)

    print('Epoch [{}/{}], Loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, mae: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss, avg_val_acc, mae))

    if avg_val_acc > best_accuracy:
        best_accuracy = avg_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print('It is best acc')

model.load_state_dict(best_model_wts)

# Plotting loss, accuracy and MAE
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 8))
ax1.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
ax1.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
ax1.legend()
ax2.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
ax2.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
ax2.legend()
ax3.plot(range(num_epochs), mae_list, color='red', linestyle='-', label='mae')
ax3.legend()
plt.show()
