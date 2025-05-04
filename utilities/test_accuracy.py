import csv
import json

from tqdm import tqdm

from categories.accuracy import *

try:
    with open("../data/translations.json", "r") as f:
        translations = json.loads(f.read())
except Exception as e:
    print(e)
    translations = None

accuracy_scores = []
print("Calculating accuracy scores...")
for t in tqdm(translations):
    acc_s = accuracy(t["german"], t["english"])
    accuracy_scores.append(acc_s["score"])

# Create a CSV file
with open("accuracy_scores.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["German", "English", "Accuracy Score"])
    # Write the data
    print("\nWriting to CSV...")
    for i, t in tqdm(enumerate(translations)):
        writer.writerow([t["german"], t["english"], accuracy_scores[i]])

print(f"CSV file created with {len(translations)} entries.")
