
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def random_cluster(pr,cluster_path): # return pr of the cluster samples randomly
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']

    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * pr ) 
        pool = torch.rand(len(cluster_sample_idx))
        index = pool.topk(num_samples)[1]
        all_samples_idx.extend(cluster_sample_idx[index])
    return all_samples_idx


#return either nearest or furthest clusters
def furthestmid_cluster(pr, cluster_path, largest=True): # return pr of the cluster samples randomly
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]        
    feat = res['x_org']
    if feat.shape[0] == 1:
        feat = feat[0]
    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * (pr) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=largest)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx


#drop clusters randomly
def drop_clusters(pr, cluster_path='clusters/cluster_clip_24.pth', seed=42):  # fixed seed added
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]

    feat = res['x_org']


    if feat.shape[0] == 1:
        feat = feat[0]


    cluster_ids = torch.randperm(num_cl).tolist()

    # Select clusters to keep
    num_keep = int(num_cl * pr)
    keep_clusters = set(cluster_ids[:num_keep])
    print('***************Keeping clusters:', keep_clusters)

    all_samples_idx = []
    for l in range(num_cl):
        if l not in keep_clusters:
            continue
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * pr)

        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=False)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx
# return an equal number of samples. If a cluster contains less samples,
#take the remaining samples from the remaning clusters equally.
#if pr=-1, the number is determined by the smallest cluster.
def equal_cluster(cluster_path, pr=-1, largest=True, is_random=False): 
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]
    print('**feat shape', feat.shape)

    all_samples_idx = []
    min_size = 1000000
    total_samples = 0
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        total_samples += len(cluster_sample_idx)
        if len(cluster_sample_idx) < min_size:
            min_size = len(cluster_sample_idx)
    print('***min size cluster', min_size)
    print('pr is ', pr)
    if pr > 0:
        samples_per_cluster = int((total_samples/num_cl)*pr)
        print('***samples per cluster', samples_per_cluster)
        repeat = 1
        total_cumul = samples_per_cluster
        while (repeat < 2):
            repeat += 1
            cumulative = 0
            for l in range(num_cl):
                cluster_sample_idx = torch.where(labels == l)[0]
                if len(cluster_sample_idx) < samples_per_cluster:
                    cumulative += samples_per_cluster - len(cluster_sample_idx) 
            total_cumul += cumulative//num_cl
            print('***total_cumul', total_cumul)
            break
        samples_per_cluster = total_cumul
    else:
        samples_per_cluster = min_size
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = min_size
        if pr > 0:
            num_samples = samples_per_cluster
        if not is_random:
            dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
            index = dist.topk(min(num_samples,len(cluster_sample_idx)) , largest=largest)[1]
        else:
            pool = torch.rand(len(cluster_sample_idx))
            index = pool.topk(min(num_samples,len(cluster_sample_idx)))[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx


