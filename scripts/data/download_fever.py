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

# Download train
r = requests.get(train_url, allow_redirects=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + "/train.jsonl", "wb") as file:
    file.write(r.content)

# Download Wikipedia
r = requests.get(train_url, allow_redirects=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(save_dir + "/wikidump")
