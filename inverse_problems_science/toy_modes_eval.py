import torch
import numpy as np
import matplotlib.pyplot as plt

import toy_modes_train
import config as c

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

i_blocks = 3
c.N_blocks = i_blocks
if i_blocks == 1:
    c.exponent_clamping = 8.

import model

model.load(F'output/toy_modes_test_{i_blocks}.pt_2')

def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    for v in dir(c):
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"

    print(config_str)

print_config()

def concatenate_test_set():
    x_all, y_all = [], []

    for x,y in c.test_loader:
        x_all.append(x)
        y_all.append(y)

    return torch.cat(x_all, 0), torch.cat(y_all, 0)

x_all, y_all = concatenate_test_set()

print(np.shape(x_all), np.shape(y_all))

# idx = [0, 1]
# x, y = x_all[idx, :], y_all[idx, :]
x_all, y_all = x_all.to(c.device), y_all.to(c.device)

print('x shape', x_all.size())

def noise_batch(ndim):
    # return torch.randn(c.batch_size, ndim).to(c.device)
    return torch.randn(x_all.size()[0], ndim).to(c.device)

fwd_input = torch.cat((x_all, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)

yz_hat = model.model(fwd_input)
y_hat = yz_hat[:, -c.ndim_y:]


class_est = torch.argmax(y_hat, dim=1).cpu().numpy()
class_gt = torch.argmax(y_all, dim=1).cpu().numpy()

cm = confusion_matrix(class_gt, class_est)
acc = accuracy_score(class_gt, class_est)


score_prec, score_reca = precision_score(class_gt, class_est, average='micro'), recall_score(class_gt, class_est, average='micro')


print("acc = {:.2f}, precision = {:.2f}, recall = {:.2f}"
      .format(100*acc, 100*score_prec, 100*score_reca))

print("confusion mat:")
print(cm)



# if c.ndim_pad_x:
#     x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)

# y_hat = model.model(x)

# print(np.shape(x), np.shape(y))

# def sample_posteriors():

#     rev_inputs = torch.cat([torch.randn(y_all.shape[0], c.ndim_z), y_all + 0.01 * torch.randn(y_all.shape)], 1).to(c.device)

#     with torch.no_grad():
#         x_samples =  model.model(rev_inputs, rev=True)
#         print(x_samples[0,:])
#         print(x_samples.size())

#     x_samples = x_samples.cpu().numpy()
#     values = torch.mm(y_all, torch.Tensor([np.arange(8)]).t()).numpy()
    
    # plt.figure(figsize=(8,8))
    # plt.scatter(x_samples[:,0], x_samples[:,1], c=values.flatten(), cmap='Set1', s=2., vmin=0, vmax=9)
    # plt.axis('equal')
    # plt.axis([-3,3,-3,3])

# sample_posteriors()
# plt.tight_layout()
# plt.savefig(F'ablation_{i_blocks}.png')
# plt.show()

