import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import PIL
from PIL import Image


train_data_path = "Q:\\train"
val_data_path = "Q:\\val"
test_data_path = "Q:\\test"


class Dataset(object):
	def __getitem__(self, index):
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError

transforms = transforms.Compose([transforms.Resize(64),	transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transforms)
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms)

batch_size=64

train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)



class CNN(nn.Module):
	def __init__(self,x, num_classes=2):

		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, num_classes)
		)
	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

y=torch.rand(2)
simpl = CNN(y)


optimizer = optim.Adam(simpl.parameters(), lr=0.001)

training_loss =0.0
valid_loss =0.0

def train(model, optimizer, loss_fn, train_loader, val_loader, device="cpu"):
	epochs = 600
	train_iterator =0
	valid_iterator =0
	for epoch in range(epochs):
		training_loss = 0.0
		valid_loss = 0.0
		model.train()
		for batch in train_loader:
			optimizer.zero_grad()
			inputs, targets = batch
			inputs = inputs.to(device)
			targets = targets.to(device)
			output = model(inputs)
			loss = loss_fn(output, targets)
			loss.backward()
			optimizer.step()
			training_loss += loss.data.item()
			train_iterator+=1
		training_loss /= train_iterator

		model.eval()
		num_correct = 0
		num_examples = 0
		for batch in val_loader:
			inputs, targets = batch
			inputs = inputs.to(device)
			output = model(inputs)
			targets = targets.to(device)
			loss = loss_fn(output,targets)
			valid_loss += loss.data.item()
			correct = torch.eq(torch.max(F.log_softmax(output,1),1)[1], targets).view(-1)
			num_correct += torch.sum(correct).item()
			num_examples += correct.shape[0]
			valid_iterator+=1
		valid_loss /= valid_iterator
		print('Epoch: {}, Training Loss: {:.10f}, Validation Loss: {:.10f}, accuracy = {:.2f}'.format(epoch+1, training_loss, valid_loss, num_correct / num_examples))


device ='cpu'
train(simpl, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, test_data_loader, device)


labels = ['volk','zaec']

img1 = Image.open('D:\\volchara.jpg')			 
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

img2 = Image.open('D:\\zaeca.jpg')			 
img2 = transforms(img2)
img2 = img2.unsqueeze(0)


#volk
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#zaec
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])


'''
##########################################
img1 = Image.open('D:\\Krug.jpg')			 
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

img2 = Image.open('D:\\Treugolnik.jpg')			 
img2 = transforms(img2)
img2 = img2.unsqueeze(0)

img3 = Image.open('D:\\O.jpg')			 
img3 = transforms(img3)
img3 = img3.unsqueeze(0)

img4 = Image.open('D:\\TR.jpg')			 
img4 = transforms(img4)
img4 = img4.unsqueeze(0)

img6 = Image.open('D:\\triangle.jpg')			 
img6 = transforms(img6)
img6 = img6.unsqueeze(0)

img7 = Image.open('D:\\kvadrat.jpg')			 
img7 = transforms(img7)
img7 = img7.unsqueeze(0)

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#triangle
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

#O
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])

#TR
prediction = simpl(img4)
prediction = prediction.argmax()
print(labels[prediction])

#triangle
prediction = simpl(img6)
prediction = prediction.argmax()
print(labels[prediction])

#kvadrat
prediction = simpl(img7)
prediction = prediction.argmax()
print(labels[prediction])
'''