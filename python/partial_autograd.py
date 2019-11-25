#!/usr/bin/env python

import torch

w = torch.tensor([0.1,0.2,0.3,0.4,0.5]).float()
x = torch.tensor([1,2,3,4,5]).float()
w.requires_grad = True
x.requires_grad = True

wx = torch.mm(w.view(1, 5), x.view(5, 1)).view(1)
wx.retain_grad()
p = torch.sigmoid(wx)
y = torch.ones(1)
loss_fn = torch.nn.BCELoss()
loss = loss_fn(p, y)
loss.backward()
print(w.grad)
print(x.grad)
print(wx.grad)

grad_input = wx.grad

w = torch.tensor([0.1,0.2,0.3,0.4,0.5]).float()
x = torch.tensor([1,2,3,4,5]).float()
w.requires_grad = True
x.requires_grad = True

wx = torch.mm(w.view(1, 5), x.view(5, 1)).view(1)
p = torch.sigmoid(wx)
y = torch.ones(1)
loss_fn = torch.nn.BCELoss()
loss = loss_fn(p, y)
# loss.backward()
# print(w.grad)
# print(x.grad)
# print(wx.grad)
wx.backward(grad_input)
print(w.grad, x.grad)