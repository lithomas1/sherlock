download = curl
py = python3
clientf = sherlock/demos/client

all: client

requirements-loose.txt:
	$(py) -m pip install -r requirements-loose.txt
	$(py) -c "import nltk; nltk.download('punkt')"

test: dataset requirements-loose.txt
	$(py) -m pytest tests

format:
	$(py) -m isort .
	$(py) -m black .

dataset: data/fever/train.jsonl data/fever/wikidump

host:
	lt --port 8080 --subdomain sherlock

data/fever/train.jsonl:
	mkdir -p data/fever
	$(download) -o data/fever/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl

data/fever/wikidump:
	mkdir -p data/fever
	$(download) -o data/fever/wikidump.zip https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
	unzip data/fever/wikidump.zip -d data/fever/wikidump
	rm -rf data/fever/wikidump.zip

client: ${clientf}/public/build/.built

sherlock/demos/client/node_modules: ${clientf}/package.json
	cd ${clientf} && npm i

sherlock/demos/client/public/build/.built: ${clientf}/node_modules ${clientf}/src/*
	cd ${clientf} && DEV=1 npm run build
	@touch ${clientf}/public/build/.built

.PHONY: clean
clean:
	rm -rf data/fever
