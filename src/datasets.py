import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


DATASET_ALIASES = {
    "iapr": "iapr",
    "iapr-tc12": "iapr",
    "mirflickr25k": "mirflickr25k",
    "mirflickr": "mirflickr25k",
    "mscoco": "mscoco",
    "ms-coco": "mscoco",
    "nus_wide_tc10": "nus_wide_tc10",
    "nus-wide-tc10": "nus_wide_tc10",
}

PROCESSED_DATASET_DIRS = {
    "mirflickr25k": "mirflickr25k",
    "iapr": "iapr_tc12",
    "mscoco": "mscoco",
    "nus_wide_tc10": "nuswide_tc10",
}

# Keep the historical split sizes so reorganizing the storage layout does not
# silently change the train/test partition that earlier experiments used.
DEFAULT_SPLIT_POLICY = {
    "mirflickr25k": {"test_size": 2000, "test_from_head": False},
    "iapr": {"test_size": 2000, "test_from_head": True},
    "mscoco": {"test_size": 5000, "test_from_head": True},
    "nus_wide_tc10": {"test_size": 2100, "test_from_head": False},
}


class ImagePathSampler:
    def __init__(self, root, paths):
        self.root = Path(root)
        self.paths = np.asarray(paths).reshape(-1)

    def __getitem__(self, index):
        path = self.paths[index]
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        return Image.open(self.root / str(path)).convert("RGB")

    def __len__(self):
        return len(self.paths)


def text_transform(text):
    return text


def normalize_dataset_name(name):
    normalized = name.strip().lower()
    if normalized not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[normalized]


def resolve_dataset_base(data_name, data_root):
    if data_root:
        return Path(data_root).expanduser()

    env_keys = [
        f"HRCH_{normalize_dataset_name(data_name).upper()}_ROOT",
        "HRCH_DATA_ROOT",
    ]
    for key in env_keys:
        value = os.getenv(key)
        if value:
            return Path(value).expanduser()

    raise ValueError(
        "Dataset root is required. Pass --data_root or set HRCH_DATA_ROOT / "
        f"HRCH_{normalize_dataset_name(data_name).upper()}_ROOT."
    )


def first_existing_path(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve any of the expected paths:\n" + "\n".join(str(path) for path in candidates)
    )


def random_permutation(length, seed):
    generator = np.random.RandomState(seed)
    return generator.permutation(length)


def partition_indices(length, seed, test_size, partition, test_from_head=False):
    # Reproduce the old loader behavior: shuffle once with a fixed seed, then
    # slice either from the head or tail depending on the historical protocol.
    indices = random_permutation(length, seed)
    if "test" in partition.lower():
        return indices[:test_size] if test_from_head else indices[-test_size:]
    return indices[test_size:] if test_from_head else indices[:-test_size]


def build_transforms(training):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if training:
        return [
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        ]
    return [
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    ]


def has_processed_dataset(root):
    root = Path(root)
    required = ["metadata.json", "image_paths.npy", "texts.npy", "labels.npy"]
    return all((root / name).exists() for name in required)


def processed_root_candidates(base_root, data_name):
    normalized_name = normalize_dataset_name(data_name)
    dataset_dir = PROCESSED_DATASET_DIRS[normalized_name]
    return [
        Path(base_root),
        Path(base_root) / dataset_dir,
        Path(base_root) / normalized_name,
    ]


def resolve_processed_root(base_root, data_name):
    # Accept both a dataset-specific folder and the dataset root itself so the
    # loader stays compatible with slightly different deployment layouts.
    for candidate in processed_root_candidates(base_root, data_name):
        if has_processed_dataset(candidate):
            return candidate
    return None


def load_processed_dataset(root, seed, partition, data_name):
    root = Path(root)
    with open(root / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    image_root = Path(metadata["image_root"])
    if not image_root.is_absolute():
        image_root = (root / image_root).resolve()
    image_paths = np.load(root / "image_paths.npy", allow_pickle=False)
    # Feature arrays can be large, so load them through mmap to avoid copying
    # the entire dataset into RAM at construction time.
    texts = np.load(root / "texts.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")

    split_policy = DEFAULT_SPLIT_POLICY[normalize_dataset_name(data_name)]
    test_size = int(metadata.get("test_size", split_policy["test_size"]))
    test_from_head = bool(metadata.get("test_from_head", split_policy["test_from_head"]))
    indices = partition_indices(len(image_paths), seed, test_size=test_size, partition=partition, test_from_head=test_from_head)
    return ImagePathSampler(image_root, image_paths), texts, labels, indices


def load_mirflickr25k_raw(root, seed, partition):
    root = Path(root)
    image_root = first_existing_path([root / "mirflickr", root / "images", root])
    fall_path = first_existing_path(
        [
            root / "mirflickr25k-fall.mat",
            root / "FAll" / "mirflickr25k-fall.mat",
            root / "MIRFLICKR25K" / "mirflickr25k-fall.mat",
        ]
    )
    yall_path = first_existing_path(
        [
            root / "mirflickr25k-yall.mat",
            root / "YAll" / "mirflickr25k-yall.mat",
            root / "MIRFLICKR25K" / "mirflickr25k-yall.mat",
        ]
    )
    lall_path = first_existing_path(
        [
            root / "mirflickr25k-lall.mat",
            root / "LAll" / "mirflickr25k-lall.mat",
            root / "MIRFLICKR25K" / "mirflickr25k-lall.mat",
        ]
    )

    images = np.asarray(sio.loadmat(fall_path)["FAll"]).reshape(-1)
    texts = np.asarray(sio.loadmat(yall_path)["YAll"])
    labels = np.asarray(sio.loadmat(lall_path)["LAll"])
    indices = partition_indices(len(images), seed, test_size=2000, partition=partition)
    return ImagePathSampler(image_root, images), texts, labels, indices


def load_iapr_raw(root, seed, partition):
    root = Path(root)
    processed_root = first_existing_path([root / "dataprocess" / "final", root / "processed", root])
    image_root = first_existing_path([root / "final_data", root / "images", root])

    labels_data = np.load(processed_root / "labels_one_hot.npy", allow_pickle=True).item()
    text_data = np.load(processed_root / "text_vectorized_data.npy", allow_pickle=True).item()

    image_ids = np.asarray(labels_data["image_ids"]).reshape(-1)
    labels_all = np.asarray(labels_data["one_hot_labels"])
    text_image_ids = np.asarray(text_data["text_image_ids"]).reshape(-1)
    text_features = np.asarray(text_data["text_features"])
    text_by_id = {int(identifier): text_features[i] for i, identifier in enumerate(text_image_ids)}

    images, texts, labels = [], [], []
    for identifier, label in zip(image_ids, labels_all):
        image_id = int(identifier)
        image_name = f"{image_id}.jpg"
        image_path = image_root / image_name
        if not image_path.exists() or image_id not in text_by_id:
            continue
        images.append(image_name)
        texts.append(text_by_id[image_id])
        labels.append(label)

    images = np.asarray(images)
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    indices = partition_indices(len(images), seed, test_size=2000, partition=partition, test_from_head=True)
    return ImagePathSampler(image_root, images), texts, labels, indices


def load_nus_wide_raw(root, seed, partition):
    root = Path(root)
    image_root = first_existing_path([root / "Flickr", root / "images", root])
    imagelist_path = first_existing_path([root / "Imagelist.txt"])
    clean_id_path = first_existing_path([root / "clean_id.nuswide.tc10.AllTags1k.mat"])
    text_path = first_existing_path([root / "texts.nuswide.AllTags1k.mat"])
    label_path = first_existing_path([root / "labels.nuswide-tc10.mat"])

    with open(imagelist_path, "r", encoding="utf-8") as handle:
        images = np.array([line.replace("\\", "/").strip() for line in handle if line.strip()])

    clean_ids = sio.loadmat(clean_id_path)["clean_id"]
    texts = sio.loadmat(text_path)["texts"]
    labels = sio.loadmat(label_path)["labels"]
    images = np.squeeze(images[clean_ids.T])
    texts = np.squeeze(texts[clean_ids.T])
    labels = np.squeeze(labels[clean_ids.T])
    indices = partition_indices(len(images), seed, test_size=2100, partition=partition)
    return ImagePathSampler(image_root, images), texts, labels, indices


def load_mscoco_raw(root, seed, partition):
    root = Path(root)
    image_root = first_existing_path([root / "image", root / "images", root])
    labels = np.load(first_existing_path([root / "one_hot_labels.npy"]), allow_pickle=True)
    label_filenames = np.load(first_existing_path([root / "label_filenames.npy"]), allow_pickle=True)
    texts = np.load(first_existing_path([root / "text_features.npy"]), allow_pickle=True)
    text_filenames = np.load(first_existing_path([root / "text_filenames.npy"]), allow_pickle=True)

    label_filenames = [Path(name.decode("utf-8") if isinstance(name, bytes) else str(name)).stem for name in label_filenames]
    text_filenames = [Path(name.decode("utf-8") if isinstance(name, bytes) else str(name)).stem for name in text_filenames]

    common_names = sorted(set(label_filenames) & set(text_filenames))
    label_index = {name: idx for idx, name in enumerate(label_filenames) if name in common_names}
    text_index = {name: idx for idx, name in enumerate(text_filenames) if name in common_names}

    images, aligned_texts, aligned_labels = [], [], []
    for name in common_names:
        images.append(f"{name}.jpg")
        aligned_labels.append(labels[label_index[name]])
        aligned_texts.append(texts[text_index[name]])

    images = np.asarray(images)
    aligned_texts = np.asarray(aligned_texts)
    aligned_labels = np.asarray(aligned_labels)
    indices = partition_indices(len(images), seed, test_size=5000, partition=partition, test_from_head=True)
    return ImagePathSampler(image_root, images), aligned_texts, aligned_labels, indices


def load_dataset_bundle(data_name, data_root, seed, partition):
    normalized_name = normalize_dataset_name(data_name)
    base_root = resolve_dataset_base(normalized_name, data_root)
    processed_root = resolve_processed_root(base_root, normalized_name)
    if processed_root is not None:
        # Prefer the standardized dataset layout first; only fall back to raw
        # loaders for backward compatibility with older experiment folders.
        return load_processed_dataset(processed_root, seed, partition, normalized_name)

    if normalized_name == "mirflickr25k":
        return load_mirflickr25k_raw(base_root, seed, partition)
    if normalized_name == "iapr":
        return load_iapr_raw(base_root, seed, partition)
    if normalized_name == "nus_wide_tc10":
        return load_nus_wide_raw(base_root, seed, partition)
    if normalized_name == "mscoco":
        return load_mscoco_raw(base_root, seed, partition)

    raise ValueError(f"Unsupported dataset: {data_name}")


class CMDataset(data.Dataset):
    """Cross-modal dataset used by the training and evaluation pipelines."""

    def __init__(self, data_name, data_root, seed, return_index=False, partition="train"):
        self.data_name = normalize_dataset_name(data_name)
        self.partition = partition
        self.return_index = return_index
        self.transforms = build_transforms("train" in partition.lower())
        self.images, self.texts, self.labels, self.indices = load_dataset_bundle(
            data_name=self.data_name,
            data_root=data_root,
            seed=seed,
            partition=partition,
        )
        self.length = int(len(self.indices))
        self.text_dim = int(self.texts.shape[1])
        if self.length <= 0:
            raise ValueError(f"Empty partition '{partition}' for dataset '{data_name}'.")

    def _get_image(self, index):
        image = self.images[index]
        if isinstance(self.images, ImagePathSampler):
            return image

        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(2, 1, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    def __getitem__(self, index):
        sample_index = int(self.indices[index])
        image = self._get_image(sample_index)
        text = self.texts[sample_index]
        label = self.labels[sample_index]

        image_crops = [transform(image) for transform in self.transforms]
        text_features = [text_transform(text) for _ in self.transforms]

        if self.return_index:
            return index, image_crops, text_features, label
        return image_crops, text_features, label

    def __len__(self):
        return self.length
