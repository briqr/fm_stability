import torch
from torch_kmeans import KMeans
import numpy as np
import os

def main():
    
   
    root_path = 'data/celebhq/clip_features/train'
    feat_file = os.path.join(root_path, 'clip_features.pt')
    f = torch.load(feat_file)
    if isinstance(f, list):
        f = torch.stack(f)
    f = f.flatten(1)
    f = torch.from_numpy(np.asarray(f)).float()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    f = f.to(device).unsqueeze(0)
    
    print('Data shape:', f.shape)

    num_clusters = [24]
    
    for num_cl in num_clusters:
        model = KMeans(n_clusters=num_cl).to(device)
        print('******num cl %d' %num_cl)
        print('data shape', f.shape)
        res = model(f)
      
        result = dict({'labels': res.labels,  # type: ignore
        'centers':res.centers,
        'inertia': res.inertia,
        'x_org':f,
        'k': num_cl})
        for l in range(num_cl):
            print('number of points in cluster %d' %l, (res.labels==l).sum()  )

        torch.save(result, os.path.join(root_path, 'cluster_%d.pth'%(num_cl)))




if __name__ == "__main__":
    main()