import gc
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models as backbone_models
import nets
from clustering_loss.clustering_loss import compute_clustering_loss
from finetune import load_checkpoint_finetune
from src.datasets import CMDataset, normalize_dataset_name
from src.utils import ContrastiveLoss
from utils.config import parse_args


LEVEL_NAMES = ("level0", "level1", "level2", "level3")


def amp_context(enabled):
    return torch.cuda.amp.autocast(enabled=True) if enabled else nullcontext()


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)
    cudnn.benchmark = True
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            torch.set_float32_matmul_precision("high")


def ensure_output_dirs(root_dir, experiment_name, log_name):
    root = Path(root_dir)
    log_dir = root / "logs" / log_name
    checkpoint_dir = root / "ckpt" / experiment_name
    feature_dir = root / "feature" / experiment_name
    for path in (log_dir, checkpoint_dir, feature_dir):
        path.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_dir, feature_dir


def resolve_backbone_weights(args):
    if args.backbone_weights:
        path = Path(args.backbone_weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights not found: {path}")
        return path

    candidates = [
        Path(args.root_dir) / "weights" / "revcol_tiny_1k.pth",
        Path(args.root_dir) / "revcol_tiny_1k.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Backbone weights were not found. Provide --backbone_weights or place "
        "revcol_tiny_1k.pth under ./weights/."
    )


def create_data_loader(dataset, batch_size, num_workers, shuffle):
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=persistent_workers,
    )


def build_dataloaders(args):
    train_dataset = CMDataset(args.data_name, args.data_root, args.seed, return_index=True, partition="train")
    retrieval_dataset = CMDataset(args.data_name, args.data_root, args.seed, return_index=True, partition="retrieval")
    query_dataset = CMDataset(args.data_name, args.data_root, args.seed, return_index=True, partition="test")

    train_loader = create_data_loader(train_dataset, args.train_batch_size, args.num_workers, shuffle=True)
    retrieval_loader = create_data_loader(retrieval_dataset, args.eval_batch_size, args.num_workers, shuffle=False)
    query_loader = create_data_loader(query_dataset, args.eval_batch_size, args.num_workers, shuffle=False)
    return train_dataset, train_loader, retrieval_loader, query_loader


def build_models(args, text_dim, device):
    backbone = backbone_models.revcol_tiny_fintune_cluster(save_memory=True, drop_path=args.dp).to(device)
    load_checkpoint_finetune(backbone, resolve_backbone_weights(args))

    image_head = nets.ImageNet_cluster(bit=args.bit, layers=parse_int_list(args.layers), droprate=args.droprate).to(device)
    text_head = nets.TextNet_cluster(y_dim=text_dim, bit=args.bit, data_name=args.data_name, droprate=args.droprate).to(device)

    image_model = nn.Sequential(backbone, image_head).to(device)
    text_model = text_head.to(device)
    return image_model, text_model


def compute_features(data_loader, model_pair, device, level_count=4, use_amp=True):
    image_model, text_model = model_pair
    image_model.eval()
    text_model.eval()

    image_features = [[] for _ in range(level_count)]
    text_features = [[] for _ in range(level_count)]
    index_chunks = []

    for indices, images, texts, _ in tqdm(data_loader, desc="Computing features", leave=False):
        image_tensor = images[0].to(device, non_blocking=True)
        text_tensor = texts[0].to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        with torch.no_grad():
            with amp_context(use_amp):
                image_output = image_model(image_tensor.float())["hash_codes"]
                text_output = text_model(text_tensor.float())["hash_codes"]

        for level_index, level_name in enumerate(LEVEL_NAMES[:level_count]):
            image_features[level_index].append(image_output[level_name])
            text_features[level_index].append(text_output[level_name])
        index_chunks.append(indices)

    sorted_order = torch.sort(torch.cat(index_chunks))[1]
    image_feature_bank, text_feature_bank = [], []
    for level_index in range(level_count):
        image_feature_bank.append(torch.cat(image_features[level_index])[sorted_order])
        text_feature_bank.append(torch.cat(text_features[level_index])[sorted_order])
    return image_feature_bank, text_feature_bank


def sample_prototypes(features, num_clusters, seed):
    if features.size(0) < num_clusters:
        raise ValueError(f"Cannot sample {num_clusters} prototypes from {features.size(0)} features.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled_indices = torch.randperm(features.size(0), generator=generator)[:num_clusters].to(features.device)
    return features[sampled_indices]


def build_cluster_heads(args, image_feature_bank, text_feature_bank, device):
    normalized_name = normalize_dataset_name(args.data_name)
    prototype_seed = args.pseed if normalized_name == "iapr" and args.pseed != 3407 else args.seed
    cluster_sizes = parse_int_list(args.cluster_num)

    image_heads = nn.ModuleDict(
        {
            level_name: nets.cluster_mlp(
                hash_dim=args.bit,
                cluster_num=cluster_sizes[index],
                prototypes=sample_prototypes(image_feature_bank[index], cluster_sizes[index], prototype_seed),
            )
            for index, level_name in enumerate(LEVEL_NAMES)
        }
    ).to(device)
    text_heads = nn.ModuleDict(
        {
            level_name: nets.cluster_mlp(
                hash_dim=args.bit,
                cluster_num=cluster_sizes[index],
                prototypes=sample_prototypes(text_feature_bank[index], cluster_sizes[index], prototype_seed),
            )
            for index, level_name in enumerate(LEVEL_NAMES)
        }
    ).to(device)
    return image_heads, text_heads


def build_optimizer(args, modules):
    parameters = []
    for module in modules:
        parameters.extend(list(module.parameters()))

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)

    if args.ls == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], gamma=0.1)
    return parameters, optimizer, scheduler


def set_train_mode(image_model, text_model, cluster_image, cluster_text, freeze_backbone=False):
    image_model.train()
    text_model.train()
    cluster_image.train()
    cluster_text.train()
    if freeze_backbone:
        image_model[0].eval()
        image_model[0].requires_grad_(False)
    else:
        image_model[0].requires_grad_(True)


def set_eval_mode(image_model, text_model, cluster_image=None, cluster_text=None):
    image_model.eval()
    text_model.eval()
    if cluster_image is not None:
        cluster_image.eval()
    if cluster_text is not None:
        cluster_text.eval()


def train_one_epoch(
    epoch,
    args,
    train_loader,
    image_model,
    text_model,
    cluster_image,
    cluster_text,
    optimizer,
    scaler,
    criterion,
    summary_writer,
    device,
    trainable_parameters,
):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_train_mode(image_model, text_model, cluster_image, cluster_text, freeze_backbone=epoch < args.warmup_epoch)
    running_loss = 0.0
    use_amp = device.type == "cuda"

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", leave=False)
    for step, (indices, images, texts, _) in enumerate(progress):
        del indices
        image_tensor = images[0].to(device, non_blocking=True)
        text_tensor = texts[0].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp):
            image_output = image_model(image_tensor.float())
            text_output = text_model(text_tensor.float())
            retrieval_loss = criterion(image_output["hash_codes"]["level3"], text_output["hash_codes"]["level3"])
            clustering_loss = compute_clustering_loss(
                image_output=image_output,
                text_output=text_output,
                cluster_text=cluster_text,
                cluster_image=cluster_image,
                data_name=args.data_name,
                tau=args.tau,
                taup=args.taup,
                ins=args.ins,
                pro=args.pro,
                ld=args.ld,
                entroy=args.entroy,
                qua=args.qua,
            )
            loss = (1.0 - args.alpha) * retrieval_loss + args.alpha * clustering_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(trainable_parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        average_loss = running_loss / (step + 1)
        progress.set_postfix(loss=f"{average_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % args.log_interval == 0:
            global_step = epoch * len(train_loader) + step
            summary_writer.add_scalar("train/loss", average_loss, global_step)
            summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)


def collect_hash_codes(data_loader, image_model, text_model, device):
    image_codes, text_codes, labels = [], [], []
    use_amp = device.type == "cuda"
    set_eval_mode(image_model, text_model)

    with torch.no_grad():
        for _, images, texts, targets in tqdm(data_loader, desc="Encoding", leave=False):
            image_tensor = images[0].to(device, non_blocking=True)
            text_tensor = texts[0].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp_context(use_amp):
                image_output = image_model(image_tensor.float())
                text_output = text_model(text_tensor.float())

            image_codes.append(image_output["hash_codes"]["level3"])
            text_codes.append(text_output["hash_codes"]["level3"])
            labels.append(targets)

    return torch.cat(image_codes).sign_(), torch.cat(text_codes).sign_(), torch.cat(labels)


def mean_average_precision(query_codes, retrieval_codes, query_labels, retrieval_labels, k=0):
    if k == 0:
        k = retrieval_labels.shape[0]

    similarity = torch.matmul(query_codes, retrieval_codes.t())
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)[:, :k]
    relevance = (torch.matmul(query_labels.float(), retrieval_labels.t().float()) > 0).float()
    retrieved_relevance = torch.gather(relevance, 1, sorted_indices)
    precision = retrieved_relevance.cumsum(dim=1) / torch.arange(1, k + 1, device=query_codes.device)
    average_precision = (precision * retrieved_relevance).sum(dim=1)
    relevant_per_query = retrieved_relevance.sum(dim=1)
    valid_queries = relevant_per_query > 0
    if not valid_queries.any():
        return 0.0
    return (average_precision[valid_queries] / relevant_per_query[valid_queries]).mean().item()


def evaluate_train_split(image_model, text_model, retrieval_loader, device, query_size=2000):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    retrieval_images, retrieval_texts, retrieval_labels = collect_hash_codes(retrieval_loader, image_model, text_model, device)
    if query_size > 0:
        query_size = min(query_size, retrieval_labels.shape[0])
        query_images = retrieval_images[:query_size]
        query_texts = retrieval_texts[:query_size]
        query_labels = retrieval_labels[:query_size]
        retrieval_images_eval = retrieval_images[:query_size]
        retrieval_texts_eval = retrieval_texts[:query_size]
        retrieval_labels_eval = retrieval_labels[:query_size]
    else:
        query_images = retrieval_images
        query_texts = retrieval_texts
        query_labels = retrieval_labels
        retrieval_images_eval = retrieval_images
        retrieval_texts_eval = retrieval_texts
        retrieval_labels_eval = retrieval_labels

    image_to_text = mean_average_precision(query_images, retrieval_texts_eval, query_labels, retrieval_labels_eval)
    text_to_image = mean_average_precision(query_texts, retrieval_images_eval, query_labels, retrieval_labels_eval)
    return image_to_text, text_to_image, (image_to_text + text_to_image) / 2.0


def evaluate_test_split(image_model, text_model, retrieval_loader, query_loader, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    retrieval_images, retrieval_texts, retrieval_labels = collect_hash_codes(retrieval_loader, image_model, text_model, device)
    query_images, query_texts, query_labels = collect_hash_codes(query_loader, image_model, text_model, device)

    image_to_text = mean_average_precision(query_images, retrieval_texts, query_labels, retrieval_labels)
    text_to_image = mean_average_precision(query_texts, retrieval_images, query_labels, retrieval_labels)
    return image_to_text, text_to_image, (image_to_text + text_to_image) / 2.0


def checkpoint_path(checkpoint_dir, args):
    filename = f"{normalize_dataset_name(args.data_name)}_{args.arch}_{args.bit}bit_best.t7"
    return checkpoint_dir / filename


def save_checkpoint(path, epoch, best_acc, image_model, text_model, cluster_image, cluster_text, optimizer, metrics):
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "image_model_state_dict": image_model.state_dict(),
            "text_model_state_dict": text_model.state_dict(),
            "cluster_image_state_dict": cluster_image.state_dict(),
            "cluster_text_state_dict": cluster_text.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "Img2Txt": metrics[0],
            "Txt2Img": metrics[1],
            "Avg": metrics[2],
        },
        path,
    )


def maybe_load_cluster_state(module_dict, checkpoint, key, legacy_key):
    if key in checkpoint:
        module_dict.load_state_dict(checkpoint[key])
        return
    if legacy_key in checkpoint:
        for module, state in zip(module_dict.values(), checkpoint[legacy_key]):
            module.load_state_dict(state)


def load_checkpoint_if_needed(args, image_model, text_model, cluster_image, cluster_text, optimizer=None):
    if not args.resume:
        return 0, float("-inf")

    checkpoint = torch.load(args.resume, map_location="cpu")
    image_model.load_state_dict(checkpoint["image_model_state_dict"], strict=False)
    text_model.load_state_dict(checkpoint["text_model_state_dict"], strict=False)
    maybe_load_cluster_state(cluster_image, checkpoint, "cluster_image_state_dict", "cluster_mlp_image")
    maybe_load_cluster_state(cluster_text, checkpoint, "cluster_text_state_dict", "cluster_mlp_text")

    if optimizer is not None and "optimizer_state_dict" in checkpoint and not args.eval_only and not args.feature_save:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("best_acc", checkpoint.get("Avg", 0.0)))
    return start_epoch, best_acc


def load_checkpoint_state(checkpoint_path, image_model, text_model, cluster_image, cluster_text):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_model.load_state_dict(checkpoint["image_model_state_dict"], strict=False)
    text_model.load_state_dict(checkpoint["text_model_state_dict"], strict=False)
    maybe_load_cluster_state(cluster_image, checkpoint, "cluster_image_state_dict", "cluster_mlp_image")
    maybe_load_cluster_state(cluster_text, checkpoint, "cluster_text_state_dict", "cluster_mlp_text")
    return checkpoint


def export_feature_bank(args, feature_dir, retrieval_loader, image_model, text_model, device):
    set_eval_mode(image_model, text_model)
    use_amp = device.type == "cuda"

    image_storage = {level_name: [] for level_name in LEVEL_NAMES}
    text_storage = {level_name: [] for level_name in LEVEL_NAMES}
    labels = []

    with torch.no_grad():
        for _, images, texts, targets in tqdm(retrieval_loader, desc="Exporting features", leave=False):
            image_tensor = images[0].to(device, non_blocking=True)
            text_tensor = texts[0].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp_context(use_amp):
                image_output = image_model(image_tensor.float())["hash_codes"]
                text_output = text_model(text_tensor.float())["hash_codes"]

            for level_name in LEVEL_NAMES:
                image_storage[level_name].append(image_output[level_name].detach().cpu())
                text_storage[level_name].append(text_output[level_name].detach().cpu())
            labels.append(targets.detach().cpu())

    dataset_name = normalize_dataset_name(args.data_name)
    for level_name in LEVEL_NAMES:
        torch.save(torch.cat(image_storage[level_name]), feature_dir / f"{dataset_name}_{args.arch}_{args.bit}_image_{level_name}.pt")
        torch.save(torch.cat(text_storage[level_name]), feature_dir / f"{dataset_name}_{args.arch}_{args.bit}_text_{level_name}.pt")
    torch.save(torch.cat(labels), feature_dir / f"{dataset_name}_{args.arch}_{args.bit}_labels.pt")


def main():
    args = parse_args()
    prepare_environment(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, checkpoint_dir, feature_dir = ensure_output_dirs(args.root_dir, args.pretrain_dir, args.log_name)

    train_dataset, train_loader, retrieval_loader, query_loader = build_dataloaders(args)
    image_model, text_model = build_models(args, text_dim=train_dataset.text_dim, device=device)

    image_feature_bank, text_feature_bank = compute_features(
        data_loader=retrieval_loader,
        model_pair=(image_model, text_model),
        device=device,
        use_amp=device.type == "cuda",
    )
    cluster_image, cluster_text = build_cluster_heads(args, image_feature_bank, text_feature_bank, device)
    trainable_parameters, optimizer, scheduler = build_optimizer(args, [image_model, text_model, cluster_image, cluster_text])

    start_epoch, best_acc = load_checkpoint_if_needed(args, image_model, text_model, cluster_image, cluster_text, optimizer=optimizer)
    criterion = ContrastiveLoss(args.margin, shift=args.shift)
    scaler = GradScaler(enabled=device.type == "cuda")
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    if args.feature_save:
        if not args.resume:
            raise ValueError("--feature_save requires --resume to load a trained checkpoint.")
        export_feature_bank(args, feature_dir, retrieval_loader, image_model, text_model, device)
        summary_writer.close()
        return

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume to load a trained checkpoint.")
        image_to_text, text_to_image, avg = evaluate_test_split(
            image_model=image_model,
            text_model=text_model,
            retrieval_loader=retrieval_loader,
            query_loader=query_loader,
            device=device,
        )
        print(f"Evaluation\nImg2Txt: {image_to_text:.6f}\tTxt2Img: {text_to_image:.6f}\tAvg: {avg:.6f}")
        summary_writer.close()
        return

    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    best_checkpoint = checkpoint_path(checkpoint_dir, args)
    epochs_without_improvement = 0
    decline_streak = 0
    previous_train_avg = None

    for epoch in range(start_epoch, args.max_epochs):
        train_one_epoch(
            epoch=epoch,
            args=args,
            train_loader=train_loader,
            image_model=image_model,
            text_model=text_model,
            cluster_image=cluster_image,
            cluster_text=cluster_text,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            summary_writer=summary_writer,
            device=device,
            trainable_parameters=trainable_parameters,
        )
        scheduler.step()

        train_image_to_text, train_text_to_image, train_avg = evaluate_train_split(
            image_model=image_model,
            text_model=text_model,
            retrieval_loader=retrieval_loader,
            device=device,
            query_size=args.train_eval_query_size,
        )
        print(
            f"Epoch {epoch + 1}/{args.max_epochs} - "
            f"TrainMonitor Img2Txt: {train_image_to_text:.6f}  "
            f"TrainMonitor Txt2Img: {train_text_to_image:.6f}  "
            f"TrainMonitor Avg: {train_avg:.6f}"
        )
        summary_writer.add_scalar("monitor_train/img_to_text", train_image_to_text, epoch)
        summary_writer.add_scalar("monitor_train/text_to_img", train_text_to_image, epoch)
        summary_writer.add_scalar("monitor_train/avg", train_avg, epoch)

        if previous_train_avg is not None and train_avg < previous_train_avg:
            decline_streak += 1
        else:
            decline_streak = 0
        previous_train_avg = train_avg

        if train_avg > best_acc:
            best_acc = train_avg
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_checkpoint,
                epoch=epoch,
                best_acc=best_acc,
                image_model=image_model,
                text_model=text_model,
                cluster_image=cluster_image,
                cluster_text=cluster_text,
                optimizer=optimizer,
                metrics=(train_image_to_text, train_text_to_image, train_avg),
            )
        else:
            epochs_without_improvement += 1

        if decline_streak >= args.early_stop_decline_patience:
            print(
                f"Early stopping triggered: monitored train avg declined for "
                f"{decline_streak} consecutive epochs."
            )
            break

        if epochs_without_improvement >= args.early_stop_best_patience:
            print(
                f"Early stopping triggered: monitored train avg did not improve over the "
                f"best value for {epochs_without_improvement} consecutive epochs."
            )
            break

    selected_checkpoint = None
    if best_checkpoint.exists():
        selected_checkpoint = best_checkpoint
    elif args.resume:
        selected_checkpoint = Path(args.resume)

    if selected_checkpoint is not None and selected_checkpoint.exists():
        load_checkpoint_state(
            checkpoint_path=selected_checkpoint,
            image_model=image_model,
            text_model=text_model,
            cluster_image=cluster_image,
            cluster_text=cluster_text,
        )

    test_image_to_text, test_text_to_image, test_avg = evaluate_test_split(
        image_model=image_model,
        text_model=text_model,
        retrieval_loader=retrieval_loader,
        query_loader=query_loader,
        device=device,
    )
    print(
        f"Final Test (best train-monitored checkpoint) - "
        f"Img2Txt: {test_image_to_text:.6f}  "
        f"Txt2Img: {test_text_to_image:.6f}  "
        f"Avg: {test_avg:.6f}"
    )
    summary_writer.add_scalar("final_test/img_to_text", test_image_to_text)
    summary_writer.add_scalar("final_test/text_to_img", test_text_to_image)
    summary_writer.add_scalar("final_test/avg", test_avg)

    summary_writer.close()


if __name__ == "__main__":
    main()
