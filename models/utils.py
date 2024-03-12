import torch

def PCA_svd(X, k, center=True):
  n = X.size()[1]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  try:
    X_center =  torch.bmm(H[None,:,:].repeat(X.shape[0],1,1), X)
    u, s, v = torch.svd(X_center)
    features  = torch.bmm(X, v[:, :, :k])
  except:
    import pdb; pdb.set_trace()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return features
