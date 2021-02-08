"""
This script downloads the FEVER dataset(https://fever.ai/resources.html) and extracts it into the data directory

The dataset is composed of the training json file and the wikidump pages(WARNING: the wikidump pages take a lot
of space)
"""
import io
import os
import zipfile

import requests

save_dir = "../../data/fever"

train_url = "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl"
wikipedia_url = "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip"

print("Downloading training data")

# Download train
r = requests.get(train_url, allow_redirects=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + "/train.jsonl", "wb") as file:
    file.write(r.content)

print("Finished downloading training data")
print("Downloading Wikipedia articles")

# Download Wikipedia
r = requests.get(wikipedia_url, allow_redirects=True)
with zipfile.ZipFile(io.BytesIO(r.content)) as zip_file:
    zip_file.extractall(save_dir + "/wikipedia")

print("Finished downloading Wikipedia articles")

