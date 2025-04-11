import numpy as np
import torch
import os
import random as rn
import gc

from utils.config import args
from sklearn.metrics import silhouette_score

seed = args.seed
# print("===> random seed:", seed)
np.random.seed(seed)
rn.seed(seed)
#
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import warnings
warnings.filterwarnings("ignore")

from utils.config import args
from tqdm import tqdm

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

import nets as models
from utils.bar_show import progress_bar
from utils.fineture import load_checkpoint_finetune
import pdb
from src.cmdataset import CMDataset
import scipy
import scipy.spatial
import torch.nn as nn
import src.utils as utils
from src.clustering_loss import compute_clustering_loss
from torch.nn.utils.clip_grad import clip_grad_norm

import itertools
import models as models_n
from nets.cluster_mlp import cluster_mlp

device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0
start_epoch = 0
state={}
"./log/HRCH"
args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
"./ckpt/HRCH"
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)
"./feature/HRCH"
args.feature_dir = os.path.join(args.root_dir, 'feature', args.pretrain_dir)
"./revcol_tiny_1k.pth"
path_pth_tiny = os.path.join(args.root_dir, args.reversible_path)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.feature_dir, exist_ok=True)

def get_list(cluster_num: str):
    cluster_num_s = cluster_num.split(",")
    cluster_num_i = [int(i) for i in cluster_num_s]
    return cluster_num_i

def get_prototype(features, num_cluster):
    prototypes_indices = np.random.choice(features.shape[0], num_cluster, replace=False)
    prototypes = features[prototypes_indices]
    return prototypes

def compute_features(train_loader, model: list, level=4, size=[args.bit] * 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_all_image = [torch.zeros(len(train_loader.dataset), i).to(device) for i in size]
    features_all_text = [torch.zeros(len(train_loader.dataset), i).to(device) for i in size]

    for batch_idx, (idx, images, texts, _) in tqdm(enumerate(train_loader),
                                                   total=len(train_loader),
                                                   desc='Computing features'):
        images = [img.to(device) for img in images]
        texts = [text.to(device) for text in texts]

        with torch.no_grad():
            model[0].eval()
            model[1].eval()
            feat_img = model[0](images[0].float())["hash_codes"]
            feat_text = model[1](texts[0].float())["hash_codes"]
            for lvl in range(level):
                features_all_image[lvl][idx, :] = feat_img[f"level{lvl}"]
                features_all_text[lvl][idx, :] = feat_text[f"level{lvl}"]

    return features_all_image, features_all_text

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = dist.shape[1]
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def gram_schmidt(vectors):
    orthogonal_vectors = []
    for v in vectors:
        for u in orthogonal_vectors:
            v -= torch.dot(v, u) / torch.dot(u, u) * u
        orthogonal_vectors.append(v / v.norm())
    return torch.stack(orthogonal_vectors)

def train(epoch):
    global train_loader, image_model, text_model, cluster_text, cluster_image, optimizer, parameters, summary_writer
    global criterion_contrast,contrast
    gc.collect()
    torch.cuda.empty_cache()
    print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
    train_loss, correct, total = 0., 0., 0.
    set_train(epoch < args.warmup_epoch)

    for batch_idx, (idx, images, texts, _) in enumerate(train_loader):

        images, texts, idx = [img.cuda() for img in images], [text.cuda() for text in texts], [idx.cuda()]
        images_outputs = [image_model(im.float()) for im in images]
        texts_outputs = [text_model(txt.float()) for txt in texts]
        images_output_level3 = images_outputs[0]["hash_codes"]["level3"]
        texts_output_level3 = texts_outputs[0]["hash_codes"]["level3"]

        clustering_loss = compute_clustering_loss(image_output=images_outputs[0],
                                                            text_output=texts_outputs[0],
                                                            cluster_text=cluster_text,
                                                            cluster_image=cluster_image,
                                                            tau=args.tau,
                                                            ins=args.ins,
                                                            pro=args.pro,
                                                            ld=args.ld,
                                                            entroy=args.entroy,
                                                            qua=args.qua,
                                                    beta=args.beta)


        Lr = criterion(images_output_level3, texts_output_level3)

        loss = (1 - args.alpha) * Lr + args.alpha * clustering_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(parameters, 1.)
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                     % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

        if batch_idx % args.log_interval == 0:
            summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1),
                                      epoch * len(train_loader) + batch_idx)
            summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'],
                                      epoch * len(train_loader) + batch_idx)

def eval(data_loader):
    imgs, txts, labs = [], [], []
    with torch.no_grad():
        for batch_idx, (_, images, texts, targets) in enumerate(data_loader):
            images, texts, targets = [img.cuda() for img in images], [text.cuda() for text in texts], targets.cuda()

            images_outputs = [image_model(im.float()) for im in images]
            texts_outputs = [text_model(txt.float()) for txt in texts]

            images_output_level3 = [images_outputs[0]["hash_codes"]["level3"]]
            texts_output_level3 = [texts_outputs[0]["hash_codes"]["level3"]]

            imgs += images_output_level3
            txts += texts_output_level3
            labs.append(targets)

        imgs = torch.cat(imgs).sign_().cpu().numpy()
        txts = torch.cat(txts).sign_().cpu().numpy()
        labs = torch.cat(labs).cpu().numpy()
    return imgs, txts, labs

def test(epoch, is_eval=True):
    global retrieval_loader, query_loader, image_model, text_model
    gc.collect()
    torch.cuda.empty_cache()
    global best_acc
    set_eval()
    (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader)
    if is_eval:
        query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
        retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
    else:
        (query_imgs, query_txts, query_labs) = eval(query_loader)

    i2t = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
    t2i = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='hamming')

    avg = (i2t + t2i) / 2.

    print(
        '%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test', i2t, t2i, (i2t + t2i) / 2.))

    if epoch and avg > best_acc :
        print('Saving......')
        state = {
            'image_model_state_dict': image_model.state_dict(),
            'text_model_state_dict': text_model.state_dict(),
            "cluster_mlp_image":[cluster.state_dict() for cluster in cluster_image.values()],
            "cluster_mlp_text": [cluster.state_dict() for cluster in cluster_text.values()],
            'optimizer_state_dict': optimizer.state_dict(),
            'Avg': avg,
            'Img2Txt': i2t,
            'Txt2Img': t2i,
            'epoch': epoch,
        }
        best_acc = avg
        torch.save(state,os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % (args.data_name, args.arch, args.bit)))

    return i2t, t2i

def set_train(is_warmup=False):
    image_model.train()

    if is_warmup and backbone:
        backbone.eval()
        backbone.requires_grad_(False)
        fea_net.train()
    elif backbone:
        backbone.requires_grad_(True)
        fea_net.train()
    text_model.train()
    for cluster in cluster_text.values():
        cluster.train()
    for cluster in cluster_image.values():
        cluster.train()

def set_eval():
    image_model.eval()
    text_model.eval()
    if args.resume :
        pass
    else:
        for cluster in cluster_text.values():
            cluster.eval()
        for cluster in cluster_image.values():
            cluster.eval()

def main():
    global train_loader, retrieval_loader, query_loader, cluster_loader
    global image_model, text_model, backbone, fea_net
    global cluster_text, cluster_image, parameters
    global optimizer, lr_schedu, summary_writer
    global criterion
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print('===> Preparing data ..')
    # build data
    train_dataset = CMDataset(
        args.data_name,
        return_index=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    # print("reset the random seed of numpy:", seed)
    np.random.seed(seed)
    cluster_dataset = CMDataset(
        args.data_name,
        partition='cluster',
        return_index=True
    )
    cluster_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # print("reset the random seed of numpy:", seed)
    np.random.seed(seed)
    retrieval_dataset = CMDataset(
        args.data_name,
        partition='retrieval',
        return_index=True
    )

    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # print("reset the random seed of numpy:", seed)
    np.random.seed(seed)
    test_dataset = CMDataset(
        args.data_name,
        partition='test',
        return_index=True
    )

    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print('===> Building REVC..')

    global image_model, text_model, backbone, fea_net
    global cluster_text, cluster_image, parameters
    global optimizer, lr_schedu, summary_writer

    backbone = models_n.revcol_tiny_fintune_cluster(save_memory=True, drop_path=args.drop_path).cuda()
    load_checkpoint_finetune(model=backbone, path=path_pth_tiny)
    fea_net = models.ImageNet_cluster(bit=args.bit, layers=get_list(args.layers),droprate=args.droprate).cuda()
    image_model = nn.Sequential(backbone, fea_net)
    text_model = models.TextNet_cluster(y_dim=train_dataset.text_dim, bit=args.bit,droprate=args.droprate).cuda()

    if args.resume:
        print("===> resuming")
        test_path=args.test_path
        chp=torch.load(os.path.join(args.ckpt_dir,test_path))
        image_model.load_state_dict(chp['image_model_state_dict'])
        text_model.load_state_dict(chp['text_model_state_dict'])

    else:
        num_cluster = get_list(args.cluster_num)
        features_all_image, features_all_text = compute_features(
            train_loader=cluster_loader,
            model=[image_model, text_model],
            size=[args.bit] * 4,
            level=4
        )
        cluster_text = {
            "level0": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[0],
                                  prototypes=get_prototype(features_all_text[0], num_cluster[0])).cuda(),
            "level1": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[1],
                                  prototypes=get_prototype(features_all_text[1], num_cluster[1])).cuda(),
            "level2": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[2],
                                  prototypes=get_prototype(features_all_text[2], num_cluster[2])).cuda(),
            "level3": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[3],
                                  prototypes=get_prototype(features_all_text[3], num_cluster[3])).cuda()
        }
        cluster_image = {
            "level0": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[0],
                                  prototypes=get_prototype(features_all_image[0], num_cluster[0])).cuda(),
            "level1": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[1],
                                  prototypes=get_prototype(features_all_image[1], num_cluster[1])).cuda(),
            "level2": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[2],
                                  prototypes=get_prototype(features_all_image[2], num_cluster[2])).cuda(),
            "level3": cluster_mlp(hash_dim=args.bit, cluster_num=num_cluster[3],
                                  prototypes=get_prototype(features_all_image[3], num_cluster[3])).cuda()
        }
        parameters = (
            list(image_model.parameters()) +
            list(text_model.parameters()) +
            list(itertools.chain.from_iterable([cluster.parameters() for cluster in cluster_image.values()])) +
            list(itertools.chain.from_iterable([cluster.parameters() for cluster in cluster_text.values()]))
        )

        wd = args.wd
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
        if args.ls == 'cos':
            lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
        else:
            lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], gamma=0.1)

        lr_schedu.step(start_epoch)

    summary_writer = SummaryWriter(args.log_dir)

    global criterion
    criterion = utils.ContrastiveLoss(args.margin, shift=args.shift)

    gc.collect()
    torch.cuda.empty_cache()

    if args.resume:
        i2t, t2i= test(epoch=False, is_eval=False)

    else:
        for epoch in range(start_epoch, args.max_epochs):
            train(epoch)
            lr_schedu.step()
            i2t, t2i = test(epoch,is_eval=True)
        chp = torch.load(os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % (args.data_name, args.arch, args.bit)))
        image_model.load_state_dict(chp['image_model_state_dict'])
        text_model.load_state_dict(chp['text_model_state_dict'])
        i2t, t2i= test(epoch=False, is_eval=False)
        summary_writer.close()

if __name__ == '__main__':
    main()
