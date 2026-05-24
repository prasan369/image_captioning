import torch
import torch.nn as nn
from dataset import get_dataloader
from model import ImageCaptioner
import time


# ── SETTINGS ─────────────────────────────────────────────

IMAGES_DIR = '/content/flickrdataset/Images'
CAPTIONS_FILE = '/content/flickrdataset/captions.txt'

EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4

SAVE_PATH = '/content/model.pth'


# ── TRAINING ─────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # FIX: pin_memory is now handled inside dataset file
    dataloader, vocab = get_dataloader(
        IMAGES_DIR,
        CAPTIONS_FILE,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size} | Batches per epoch: {len(dataloader)}")

    model = ImageCaptioner(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    torch.backends.cudnn.benchmark = True

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (images, captions) in enumerate(dataloader):

            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]

            outputs = outputs[:, :targets.shape[1], :]

            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1} done")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f} sec\n")

    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab
    }, SAVE_PATH)

    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()