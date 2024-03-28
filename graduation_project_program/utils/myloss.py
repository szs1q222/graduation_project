import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义loss（此处等价于nn.CrossEntropyLoss）
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, label, eps=1e-3):
        '''
        前向传播
        交叉熵返回 -y实际*log(y预测)，y实际只有一个为1，则返回 -log(y预测)
        :param pred:预测值，model.forward返回的值self.result(x)
        :param label:标签，ReadYOLO.__getitem__返回的target，就是torch.from_numpy(array_target).to(self.device)，一个tensor其中每个值为图片的标签
        :param eps:eps为了防止log无限小（梯度爆炸）
        :return:
        '''
        batch_size = pred.shape[0]  # shape获取维度信息，shape[i]读取矩阵第i维的长度，此处获取batch维度
        new_pred = pred.reshape(batch_size, -1)  # 将pred的batch维度后的维度合并
        expand_target = torch.zeros(new_pred.shape, device=device)  # 扩充target，建立一个和new_pred大小一致的全0张量
        for i in range(batch_size):
            expand_target[i, int(label[i])] = 1  # 对应batch的图片对应类别赋值为1，形成one-hot矩阵
        # 对预测输出进行softmax（0<y预测<1，防止y过大的梯度爆炸，eps防止y过小的梯度爆炸），softmax也可能出现NaN
        softmax_pred = torch.softmax(new_pred, dim=1)
        # 其中一种预防方式,softmax:= np.exp(self.H - np.max(self.H)) / np.sum(np.exp(self.H - np.max(self.H)), axis=0)
        return torch.sum(-torch.log(softmax_pred + eps) * expand_target) / batch_size  # 计算总损失

    # 继承nn.Module重写forward，调用时可以直接括号赋值，可以不用写__call__
    def __call__(self, *args, **kwargs):
        return self.forward(*args)


# 测试
if __name__ == '__main__':
    a = torch.tensor([[0.0791, -0.2797, 0.5169, -0.1229, 0.4389],
                      [-0.1366, 0.0622, 0.1356, 0.2859, 0.5595]], device=device)
    b = torch.tensor([[0], [1]], device=device)
    loss = MyLoss()
    myloss = loss(a, b)
    print(myloss)
