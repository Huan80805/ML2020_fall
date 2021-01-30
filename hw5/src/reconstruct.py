#for problem2
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import preprocess
from autoencoder import AE
from test import inference,predict

def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)

# 畫出原圖
trainX = np.load("trainX.npy")
plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# 畫出 reconstruct 的圖
model = AE().cuda()
model.load_state_dict(torch.load('checkpoints/last_checkpoint.pth'))
model.eval()
train_X_preprocess = preprocess(trainX)
inp = torch.Tensor(train_X_preprocess[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.tight_layout()

#二維平面上視覺化label的分佈
trainY = np.load('trainY.npy')
latents = inference(trainX, model)
pred_from_latent, emb_from_latent = predict(latents)
acc_latent = cal_acc(trainY, pred_from_latent)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(emb_from_latent, trainY, savefig='baseline.png')
