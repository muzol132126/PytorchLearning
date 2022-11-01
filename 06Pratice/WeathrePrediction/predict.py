import pandas as pd
import numpy as np
import datetime
from torch import nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
import time

features = pd.read_csv('06Pratice/WeathrePrediction/temps.csv')

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# print(dates[:5])

features = pd.get_dummies(features)

# 标签
labels = np.array(features['actual'])
# 在特征中去掉标签
features = features.drop('actual', axis = 1)

# 名字单独保存一下
feature_list = list(features.columns)

# 转换成合适的格式
features = np.array(features)
# print(features[:5])
input_features = preprocessing.StandardScaler().fit_transform(features)
# transform = torchvision.transforms.Compose(torchvision.transforms.ToTensor())


class WeatherPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.double),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size, dtype=torch.double),
            )

    def forward(self, x):
        x = self.model(x)
        return x


input_size = input_features.shape[1] # shape[0]取行数,shape[1]取列数(特征数)
hidden_size = 128
output_size = 1
batch_size = 16

model = WeatherPrediction(input_size, hidden_size, output_size)

learning_rate = 0.001
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
start_time = time.time()
epoch = 100
for i in range(epoch):
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        x = torch.tensor(input_features[start:end], dtype=torch.double)    
        y = torch.tensor(labels[start:end], dtype=torch.double)   
        prediction = model(x)
        loss = cost(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f"loss: {loss}")

       
end_time = time.time()
print(f"training time: {end_time-start_time}")

xx = torch.tensor(input_features, dtype = torch.double)
predict = model(xx).data.numpy()

dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# 再创建一个来存日期和其对应的模型预测值
predictions_data = pd.DataFrame(data = {'date': dates, 'prediction': predict.reshape(-1)}) 

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')

plt.xticks(rotation = '60'); 
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')
plt.show()