import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── VOCABULARY ─────────────────────────────────────────────

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self, captions):
        for caption in captions:
            for word in caption.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def caption_to_indices(self, caption):
        words = caption.lower().split()
        indices = [self.word2idx["<START>"]]

        for word in words:
            indices.append(self.word2idx.get(word, self.word2idx["<UNK>"]))

        indices.append(self.word2idx["<END>"])
        return indices


# ── LOAD CAPTIONS ──────────────────────────────────────────

def load_captions(captions_file):
    captions_dict = {}

    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("image"):
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            image_name, caption = parts
            image_name = image_name.split("#")[0]

            if image_name not in captions_dict:
                captions_dict[image_name] = []

            captions_dict[image_name].append(caption)

    return captions_dict


# ── DATASET ────────────────────────────────────────────────

class FlickrDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform):
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

        captions_dict = load_captions(captions_file)

        # flatten dataset
        self.samples = []
        for img, caps in captions_dict.items():
            for cap in caps:
                self.samples.append((img, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = self.vocab.caption_to_indices(caption)
        caption = torch.tensor(caption, dtype=torch.long)

        return image, caption


# ── COLLATE FUNCTION (BATCHING) ────────────────────────────

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]

    max_len = max(len(c) for c in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)

    for i, c in enumerate(captions):
        padded[i, :len(c)] = c

    return images, padded


# ── DATALOADER ─────────────────────────────────────────────

def get_dataloader(images_dir, captions_file, batch_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    captions_dict = load_captions(captions_file)

    all_captions = []
    for caps in captions_dict.values():
        all_captions.extend(caps)

    vocab = Vocabulary()
    vocab.build_vocab(all_captions)

    dataset = FlickrDataset(images_dir, captions_file, vocab, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,        # SPEED BOOST
        pin_memory=True,                # GPU transfer faster
        persistent_workers=True         # avoids reload each epoch
    )

    return dataloader, vocab


# ── TEST ───────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGES_DIR = '/content/flickrdataset/Images'
    CAPTIONS_FILE = '/content/flickrdataset/captions.txt'

    loader, vocab = get_dataloader(IMAGES_DIR, CAPTIONS_FILE, batch_size=32)

    print("Vocab size:", len(vocab))
    print("Total batches:", len(loader))

    images, captions = next(iter(loader))
    print("Image shape:", images.shape)
    print("Caption shape:", captions.shape)