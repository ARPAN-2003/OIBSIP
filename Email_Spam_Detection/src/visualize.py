import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# -----------------------------
# Create outputs folder
# -----------------------------

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin1"
)

df = df.iloc[:, 0:2]

df.columns = [
    "label",
    "message"
]

# -----------------------------
# Split Dataset
# -----------------------------

spam = df[df["label"] == "spam"]

ham = df[df["label"] == "ham"]

# -----------------------------
# Spam vs Ham Count Plot
# -----------------------------

plt.figure(figsize=(7, 5))

df["label"].value_counts().plot(
    kind="bar"
)

plt.title("Spam vs Ham Messages")

plt.xlabel("Class")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/spam_vs_ham.png"
)

plt.close()

# -----------------------------
# Message Length Distribution
# -----------------------------

df["length"] = df["message"].apply(len)

plt.figure(figsize=(8, 5))

plt.hist(
    df["length"],
    bins=40
)

plt.title("Message Length Distribution")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/message_length_distribution.png"
)

plt.close()

# -----------------------------
# Word Clouds
# -----------------------------

spam_text = " ".join(
    spam["message"]
)

ham_text = " ".join(
    ham["message"]
)

spam_cloud = WordCloud(
    width=900,
    height=500,
    background_color="black"
).generate(
    spam_text
)

ham_cloud = WordCloud(
    width=900,
    height=500,
    background_color="white"
).generate(
    ham_text
)

plt.figure(figsize=(10, 5))

plt.imshow(
    spam_cloud
)

plt.axis("off")

plt.title("Spam Word Cloud")

plt.tight_layout()

plt.savefig(
    "outputs/wordcloud_spam.png"
)

plt.close()

plt.figure(figsize=(10, 5))

plt.imshow(
    ham_cloud
)

plt.axis("off")

plt.title("Ham Word Cloud")

plt.tight_layout()

plt.savefig(
    "outputs/wordcloud_ham.png"
)

plt.close()

# -----------------------------
# Top 20 Spam Words
# -----------------------------

spam_words = []

for text in spam["message"]:

    spam_words.extend(

        str(text).lower().split()

    )

spam_counter = Counter(

    spam_words

)

top_spam = spam_counter.most_common(20)

words = [i[0] for i in top_spam]

counts = [i[1] for i in top_spam]

plt.figure(figsize=(10, 6))

plt.barh(

    words,

    counts

)

plt.title(

    "Top 20 Spam Words"

)

plt.tight_layout()

plt.savefig(

    "outputs/top20_spam_words.png"

)

plt.close()

# -----------------------------
# Top 20 Ham Words
# -----------------------------

ham_words = []

for text in ham["message"]:

    ham_words.extend(

        str(text).lower().split()

    )

ham_counter = Counter(

    ham_words

)

top_ham = ham_counter.most_common(20)

words = [i[0] for i in top_ham]

counts = [i[1] for i in top_ham]

plt.figure(figsize=(10, 6))

plt.barh(

    words,

    counts

)

plt.title(

    "Top 20 Ham Words"

)

plt.tight_layout()

plt.savefig(

    "outputs/top20_ham_words.png"

)

plt.close()

# -----------------------------
# Console Output
# -----------------------------

print()

print("=" * 50)

print("Visualization Completed Successfully")

print("=" * 50)

print()

print("Generated Files:")

print()

print("outputs/spam_vs_ham.png")

print("outputs/message_length_distribution.png")

print("outputs/wordcloud_spam.png")

print("outputs/wordcloud_ham.png")

print("outputs/top20_spam_words.png")

print("outputs/top20_ham_words.png")

print()

print("=" * 50)