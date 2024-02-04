import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# For reproductibility.
g = torch.Generator().manual_seed(2147483647)

# Read names.
words = open("family.txt", "r").read().splitlines()

# Remove blackspaces and turn
words = list(map(lambda w: w.lower().strip(), words))

# Extract characters from the text.
chars = sorted(list(set("".join(words))))

# Add + 1 for "." character -> start and end of a sequence.
number_of_characters = len(chars) + 1

# Create Loockup table.
N = torch.zeros((number_of_characters, number_of_characters), dtype=torch.int32)

# Map characters to indexes.
stoi = {
    s: i + 1 for i, s in enumerate(chars)
}  # Start at index 1, because we want '.' at 0.
stoi["."] = 0

# Map Indexes to characters.
itos = {i: s for s, i in stoi.items()}

# Create training set
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
number_of_element_in_the_dataset = xs.nelement()

# Initialize parameter for the the perception.
W = torch.randn(
    (number_of_characters, number_of_characters), generator=g, requires_grad=True
)

# Do gradient descent.
for _ in range(200):

    # Forward.
    xenc = F.one_hot(xs, num_classes=number_of_characters).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)  # Probabilities
    # Last 2 lines correspond to a SOFTMAX.
    loss = (
        -probs[torch.arange(number_of_element_in_the_dataset), ys].log().mean()
        + 0.01 * (W**2).mean()
    )
    print(loss.item())

    # Backward.
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad


# Generate 5 new name.
for _ in range(5):
    out = []
    ix = 0
    while True:

        xenc = F.one_hot(torch.tensor([ix]), num_classes=number_of_characters).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)  # Probabilities

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))
