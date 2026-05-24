import torch
import torch.nn as nn
import torchvision.models as models


# ── ENCODER (CNN) ─────────────────────────────────────────────────────────────
# Job: take an image, output a vector that summarizes what's in it
# We use a pretrained ResNet50 — it already knows how to read images
# We just remove its last layer and use the features it extracts

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        # load ResNet50 pretrained on ImageNet
        resnet = models.resnet50(pretrained=True)

        # freeze all ResNet weights — we don't want to retrain it
        # it already knows how to extract image features
        for param in resnet.parameters():
            param.requires_grad = False

        # remove the last classification layer
        # normally ResNet outputs 1000 class scores — we don't want that
        # we want the 2048 raw features before that layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # add our own linear layer to resize 2048 → embed_size
        # this connects CNN output to RNN input
        self.linear = nn.Linear(2048, embed_size)

    def forward(self, images):
        # images shape: (batch, 3, 224, 224)
        features = self.resnet(images)              # (batch, 2048, 1, 1)
        features = features.squeeze(3).squeeze(2)   # (batch, 2048)
        features = self.linear(features)            # (batch, embed_size)
        return features


# ── DECODER (RNN) ─────────────────────────────────────────────────────────────
# Job: take the image feature vector, generate a caption word by word
# The image features are fed as the first input
# Then it generates one word at a time until <END>

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()

        # embedding: converts word indices → dense vectors
        # same concept as your QA project
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM: processes the sequence and generates hidden states
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # linear: converts hidden state → scores for every word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # features shape: (batch, embed_size)  ← from encoder
        # captions shape: (batch, seq_len)     ← word indices

        # convert caption word indices → embeddings
        embeddings = self.embedding(captions)    # (batch, seq_len, embed_size)

        # prepend image features as the first "word" in the sequence
        # this is how the image information enters the LSTM
        features = features.unsqueeze(1)         # (batch, 1, embed_size)
        inputs = torch.cat((features, embeddings), dim=1)  # (batch, seq_len+1, embed_size)

        # pass through LSTM
        hidden, _ = self.lstm(inputs)            # (batch, seq_len+1, hidden_size)

        # convert to vocab scores
        outputs = self.linear(hidden)            # (batch, seq_len+1, vocab_size)
        return outputs


# ── FULL MODEL ────────────────────────────────────────────────────────────────
# Wraps Encoder + Decoder together

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        # used at inference time — no captions provided, generate from scratch
        self.eval()
        result = []
        device = image.device  # use same device as the input image

        with torch.no_grad():
            features = self.encoder(image)      # (1, embed_size)
            inputs = features.unsqueeze(1)      # (1, 1, embed_size)
            hidden = None

            for _ in range(max_length):
                out, hidden = self.decoder.lstm(inputs, hidden)
                scores = self.decoder.linear(out.squeeze(1))  # (1, vocab_size)
                predicted_idx = scores.argmax(dim=1).item()

                word = vocab.idx2word[predicted_idx]
                if word == "<END>":
                    break
                if word not in ("<START>", "<PAD>"):
                    result.append(word)

                # next input is the embedding of the predicted word
                # .to(device) ensures tensor is on same device as model
                inputs = self.decoder.embedding(
                    torch.tensor([predicted_idx]).to(device)
                ).unsqueeze(1)

        return " ".join(result)