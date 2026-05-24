import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import ImageCaptioner


# ── SETTINGS ──────────────────────────────────────────────────────────────────

MODEL_PATH  = '../model.pth'
IMAGES_DIR  = '../flickrdataset/Images'


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint (model weights + vocab we saved during training)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    # rebuild model with same settings used in training
    model = ImageCaptioner(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(vocab)
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocab, device


# ── PREPARE IMAGE ─────────────────────────────────────────────────────────────
# same transforms used during training

def prepare_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim
    return image, image_tensor


# ── GENERATE CAPTION FOR ONE IMAGE ───────────────────────────────────────────

def caption_image(image_path, model, vocab, device):
    original_image, image_tensor = prepare_image(image_path, device)
    caption = model.generate_caption(image_tensor, vocab, max_length=20)
    return original_image, caption


# ── SHOW IMAGE + CAPTION ──────────────────────────────────────────────────────

def show_result(image, caption):
    plt.imshow(image)
    plt.title(caption, fontsize=14, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ── TEST ON MULTIPLE IMAGES ───────────────────────────────────────────────────
# picks a few images from data/images and shows generated captions

def evaluate_samples(image_paths):
    model, vocab, device = load_model(MODEL_PATH)

    for image_path in image_paths:
        image, caption = caption_image(image_path, model, vocab, device)
        print(f"Image  : {image_path}")
        print(f"Caption: {caption}")
        print()
        show_result(image, caption)


# ── BLEU SCORE ────────────────────────────────────────────────────────────────
# BLEU measures how close your generated caption is to the real captions
# Score between 0 and 1 — higher is better
# Don't worry too much about this early on, visual checks are more useful

def compute_bleu(dataloader, model, vocab, device, num_batches=10):
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    references = []  # real captions
    hypotheses = []  # generated captions

    with torch.no_grad():
        for i, (images, captions) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)

            for j in range(len(images)):
                # generate caption for this image
                generated = model.generate_caption(
                    images[j].unsqueeze(0), vocab, max_length=20
                )
                generated_words = generated.split()

                # get real caption words (remove PAD, START, END)
                real_indices = captions[j].tolist()
                real_words = [
                    vocab.idx2word[idx] for idx in real_indices
                    if idx not in (0, 1, 2)  # skip PAD, START, END
                ]

                references.append([real_words])   # list of reference captions
                hypotheses.append(generated_words) # generated caption

    score = corpus_bleu(references, hypotheses)
    print(f"BLEU Score: {score:.4f}")
    return score


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # grab first 5 images from data/images to test on
    all_images = os.listdir(IMAGES_DIR)[:5]
    image_paths = [os.path.join(IMAGES_DIR, img) for img in all_images]

    evaluate_samples(image_paths)