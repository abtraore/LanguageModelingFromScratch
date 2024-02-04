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

# Populate the lookup table.
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1, ix2] += 1


# Plot the frequency table.
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(number_of_characters):
    for j in range(number_of_characters):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")

plt.savefig("frequency.png")


# Turn the Frequency table to probability.
P = (
    N + 1
).float()  # +1 is for mdel smoothing so that the log-likelihood won't get log(0) which is inf.
P /= P.sum(1, keepdim=True)

# Generate a new name.
for _ in range(5):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))


# Compute the likelihood.
log_likelihood = 0.0
count = 0

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):

        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        prob = P[ix1, ix2]

        logprob = torch.log(prob)
        log_likelihood += logprob

        count += 1
        # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")


print(f"\n'{log_likelihood=}'")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/count}")
