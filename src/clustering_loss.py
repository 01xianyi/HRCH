import numpy as np
import torch
import torch.nn.functional as F
from utils.config import args


def hierarchical_contrastive_loss(x0, x1, tau, center_index, level):
    device = x0.device
    dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()

    center_index_level = center_index
    equality_matrix = (center_index_level.unsqueeze(0) == center_index_level.unsqueeze(1))
    equality_matrix=(equality_matrix).to(torch.uint8)
    mask_fn=equality_matrix.to(device)
    logits01 = dist_f(x0, x1) / tau
    logits = logits01
    mask=mask_fn
    logits = logits - mask * 1e9
    positive = torch.diag(logits01)
    loss = torch.mean(-torch.log(
        torch.exp(positive) / (torch.exp(positive)+(torch.sum(torch.exp(logits), dim=1)))
    ))

    return loss


def contrastive_proto(hash_code,proto_mlp,center_index,tau=0.12):
    '''
    :param hash_code:
    :param proto_mlp: cluster_net
    :param center_index: label of hashcode in the other modal
    '''

    predict_matrix=F.normalize(proto_mlp(hash_code),dim=1)
    positive=predict_matrix[torch.arange(predict_matrix.size(0)), center_index.squeeze().long()]

    positive=torch.exp(positive/tau)

    all_matrix=torch.sum(torch.exp(predict_matrix/tau),dim=1)
    loss=torch.mean(-torch.log(positive/all_matrix))

    return loss

def get_label(clusternet,feature):
    cluster_result=clusternet(feature)
    _, min_indice = torch.max(cluster_result, dim=1)
    return min_indice,cluster_result

def entropy_loss(feature,alpha=1.0):
    p_mat=F.softmax(feature,dim=1)
    p_mat_mean=p_mat.mean(dim=0)
    loss=alpha*torch.sum(p_mat_mean*torch.log(p_mat_mean))
    return loss

def compute_clustering_loss(image_output, text_output ,cluster_text,cluster_image,tau=0.3,ins=0.8,pro=4,ld=4,entroy=0.01,qua=0.01,beta=0.5,taup=0.12):

    image_hash_codes = image_output['hash_codes']
    text_hash_codes = text_output['hash_codes']
    total_loss = 0

    for i, level in enumerate(["level0","level1","level2","level3"]):
        image_hash = image_hash_codes[level]
        text_hash = text_hash_codes[level]

        cluster_text_level = cluster_text[level]
        cluster_image_level = cluster_image[level]

        label_image, cluster_result_image = get_label(cluster_image_level, image_hash)
        label_text, cluster_result_text = get_label(cluster_text_level, text_hash)


        instance_contrastive_loss_image = (hierarchical_contrastive_loss(image_hash, text_hash,
                                                                       center_index=label_image,
                                                                       tau=tau,level=i)+
                                         hierarchical_contrastive_loss(text_hash, image_hash,
                                                                       center_index=label_image,
                                                                       tau=tau,level=i))/2
        instance_contrastive_loss_text = (hierarchical_contrastive_loss(image_hash, text_hash,
                                                                       center_index=label_text,
                                                                       tau=tau,level=i)+
                                         hierarchical_contrastive_loss(text_hash, image_hash,
                                                                       center_index=label_text,
                                                                       tau=tau,level=i))/2

        proto_contrastive_loss_image = (contrastive_proto(hash_code=image_hash,proto_mlp=cluster_image_level,
                                                        center_index=label_image,tau=taup)+
                                      contrastive_proto(hash_code=text_hash, proto_mlp=cluster_image_level,
                                                        center_index=label_image, tau=taup))/2
        proto_contrastive_loss_text = (contrastive_proto(hash_code=image_hash,proto_mlp=cluster_text_level,
                                                        center_index=label_text,tau=taup)+
                                      contrastive_proto(hash_code=text_hash, proto_mlp=cluster_text_level,
                                                        center_index=label_text, tau=taup))/2

        if "mir" in args.data_name.lower():
            entropy_loss_all = 0.5*(entropy_loss(cluster_result_image)+0.5*entropy_loss(cluster_result_text))
            instance_contrastive_loss=0.5*instance_contrastive_loss_image+0.5*instance_contrastive_loss_text
            proto_contrastive_loss=0.5*proto_contrastive_loss_image+0.5*proto_contrastive_loss_text

        else:
            entropy_loss_all = 0.8*(entropy_loss(cluster_result_image)+0.2*entropy_loss(cluster_result_text))
            instance_contrastive_loss=0.8*instance_contrastive_loss_image+0.2*instance_contrastive_loss_text
            proto_contrastive_loss=0.8*proto_contrastive_loss_image+0.2*proto_contrastive_loss_text

        quantization_loss = (torch.mean((torch.abs(image_hash) - torch.tensor(1.0).cuda()) ** 2) + torch.mean(
                (torch.abs(text_hash) - torch.tensor(1.0).cuda()) ** 2))

        level_loss = (ins*instance_contrastive_loss +  pro*proto_contrastive_loss) + qua * quantization_loss+entroy*entropy_loss_all
        total_loss += (1 / (i+ld)) * level_loss

    return total_loss