import torch
import torch.nn.functional as F


def hierarchical_contrastive_loss(image_hash, text_hash, tau, center_index):
    similarity = F.normalize(image_hash, dim=1) @ F.normalize(text_hash, dim=1).t()
    logits = similarity / tau
    same_center = center_index.unsqueeze(0) == center_index.unsqueeze(1)
    negative_logits = logits.masked_fill(same_center, float("-inf"))
    positive_logits = torch.diag(logits)
    return torch.mean(
        -torch.log(
            torch.exp(positive_logits)
            / (torch.exp(positive_logits) + torch.exp(negative_logits).sum(dim=1))
        )
    )


def contrastive_proto(hash_code, proto_mlp, center_index, tau=0.12):
    prototype_scores = F.normalize(proto_mlp(hash_code), dim=1)
    positive_scores = prototype_scores[torch.arange(prototype_scores.size(0)), center_index.squeeze().long()]
    positive_scores = torch.exp(positive_scores / tau)
    all_scores = torch.sum(torch.exp(prototype_scores / tau), dim=1)
    return torch.mean(-torch.log(positive_scores / all_scores))


def get_label(cluster_net, features):
    cluster_result = cluster_net(features)
    _, labels = torch.max(cluster_result, dim=1)
    return labels, cluster_result


def entropy_loss(feature, alpha=1.0):
    probabilities = F.softmax(feature, dim=1)
    probabilities_mean = probabilities.mean(dim=0).clamp_min(1e-12)
    return alpha * torch.sum(probabilities_mean * torch.log(probabilities_mean))


def compute_clustering_loss(
    image_output,
    text_output,
    cluster_text,
    cluster_image,
    data_name,
    tau=0.12,
    ins=0.8,
    pro=1.0,
    ld=4,
    entroy=0.01,
    qua=0.01,
    taup=0.12,
):
    del ld
    image_hash_codes = image_output["hash_codes"]
    text_hash_codes = text_output["hash_codes"]
    total_loss = 0.0
    is_iapr = "iapr" in data_name.lower()

    for level in ("level0", "level1", "level2", "level3"):
        image_hash = image_hash_codes[level]
        text_hash = text_hash_codes[level]

        cluster_text_level = cluster_text[level]
        cluster_image_level = cluster_image[level]

        label_image, cluster_result_image = get_label(cluster_image_level, image_hash)
        label_text, cluster_result_text = get_label(cluster_text_level, text_hash)

        instance_loss_image = (
            hierarchical_contrastive_loss(image_hash, text_hash, tau=tau, center_index=label_image)
            + hierarchical_contrastive_loss(text_hash, image_hash, tau=tau, center_index=label_image)
        ) / 2
        instance_loss_text = (
            hierarchical_contrastive_loss(image_hash, text_hash, tau=tau, center_index=label_text)
            + hierarchical_contrastive_loss(text_hash, image_hash, tau=tau, center_index=label_text)
        ) / 2

        prototype_loss_image = (
            contrastive_proto(image_hash, cluster_image_level, center_index=label_image, tau=taup)
            + contrastive_proto(text_hash, cluster_image_level, center_index=label_image, tau=taup)
        ) / 2
        prototype_loss_text = (
            contrastive_proto(image_hash, cluster_text_level, center_index=label_text, tau=taup)
            + contrastive_proto(text_hash, cluster_text_level, center_index=label_text, tau=taup)
        ) / 2

        if is_iapr:
            entropy_regularizer = 0.8 * (entropy_loss(cluster_result_image) + 0.2 * entropy_loss(cluster_result_text))
            instance_loss = 0.8 * instance_loss_image + 0.2 * instance_loss_text
            prototype_loss = 0.8 * prototype_loss_image + 0.2 * prototype_loss_text
        else:
            entropy_regularizer = 0.5 * (entropy_loss(cluster_result_image) + 0.5 * entropy_loss(cluster_result_text))
            instance_loss = 0.5 * instance_loss_image + 0.5 * instance_loss_text
            prototype_loss = 0.5 * prototype_loss_image + 0.5 * prototype_loss_text

        unit_value = image_hash.new_tensor(1.0)
        quantization_loss = (
            torch.mean((torch.abs(image_hash) - unit_value) ** 2)
            + torch.mean((torch.abs(text_hash) - unit_value) ** 2)
        )

        level_loss = ins * instance_loss + pro * prototype_loss + qua * quantization_loss + entroy * entropy_regularizer
        total_loss += 0.25 * level_loss

    return total_loss
