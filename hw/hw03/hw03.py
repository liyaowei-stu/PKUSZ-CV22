import torch

torch.manual_seed(0)

x = torch.randn(10, 4, requires_grad=True)
W = torch.randn(4, 4, requires_grad=True)
y = torch.randn(10, 4, requires_grad=True)

Z = x @ W 
m = (x @ W) >= 0

A = Z * m

L = A - y

f = (L**2).sum()

f.backward()
print("="*80)
print("###grad_W autograd: \n", W.grad, "\n")
print("###verify W_grad: 2 * x.T @ ((A-y) * (Z >= 0))  \n", 2 * x.T @ ((A-y) * (Z >= 0)))
print("="*80)
print("###grad_X autograd: \n", x.grad, "\n")
print("###verify X_grad: 2 * ((A-y) * (Z >= 0))) @ W.T  \n", 2 * ((A-y) * (Z >= 0)) @ W.T)
print("="*80)
print("###grad_Y autograd: \n", y.grad, "\n")
print("###verify W_grad: -2 * (A-y)  \n", -2 * (A-y))
