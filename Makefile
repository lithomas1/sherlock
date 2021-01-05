download = curl
py = python3

test: dataset
	$(py) -m pytest tests

format:
	$(py) -m isort .
	$(py) -m black .

dataset: data/fever/train.jsonl data/fever/wikidump

data/fever/train.jsonl:
	mkdir -p data/fever
	$(download) -o data/fever/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl

data/fever/wikidump:
	mkdir -p data/fever
	$(download) -o data/fever/wikidump.zip https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
	unzip data/fever/wikidump.zip -d data/fever/wikidump
	rm -rf data/fever/wikidump.zip

.PHONY: clean
clean:
	rm -rf data/fever
