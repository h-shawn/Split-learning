import numpy as np
import torch
import torch.utils.data as Data
from data import get_data_set

test_x, test_y, test_l = get_data_set("test")
test_x = torch.from_numpy(test_x[0:100]).float()
test_y = torch.from_numpy(test_y[0:100]).long()

model = torch.load("../model/VGG16.pkl", map_location='cpu')
device = torch.device("cpu")

test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(
    dataset=test_dataset, batch_size=128, shuffle=False)

correct_classified = 0
total = 0

for batch_num, (batch_xs, batch_ys) in enumerate(test_loader):
    outputs = model(batch_xs, 0, 0, True)
    prediction = torch.max(outputs.data, 1)
    total = total + batch_ys.size(0)
    correct_classified += np.sum(prediction[1].numpy() == batch_ys.numpy())

acc = (correct_classified/total)*100

print(
    "Accuracy on Test-Set:{0:.2f}%({1}/{2})".format(acc, correct_classified, total))
