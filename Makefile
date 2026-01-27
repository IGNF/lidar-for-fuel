# Makefile to manage main tasks
# cf. https://blog.ianpreston.ca/conda/python/bash/2020/05/13/conda_envs.html#makefile

# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
SHELL = /bin/bash


##############################
# Install
##############################
install:
	mamba env update -n lidarforfuel -f environment.yml


##############################
# Dev/Contrib tools
##############################

testing:
	python -m pytest ./test -s --log-cli-level DEBUG -m "not geopf"

testing_full:
	python -m pytest ./test -s --log-cli-level DEBUG

install-precommit:
	pre-commit install


##############################
# Build/deploy pip lib
##############################

deploy: check
	twine upload dist/*

check: dist/lidarforfuel*.tar.gz
	twine check dist/*

dist/lidarforfuel*.tar.gz:
	python -m build

build: clean
	python -m build

clean:
	rm -rf tmp
	rm -rf lidarforfuel.egg-info
	rm -rf dist

##############################
# Build/deploy Docker image
##############################

REGISTRY=ghcr.io
IMAGE_NAME=lidarforfuel
NAMESPACE=ignf
VERSION=`python -m lidarforfuel._version`
FULL_IMAGE_NAME=${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${VERSION}
DATA_DIR= `realpath ./data`


docker-build:
	docker build --no-cache -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .

docker-test:
	docker run -v ${DATA_DIR}:/lidarforfuel/data --rm -it ${IMAGE_NAME}:${VERSION} python -m pytest -s test

docker-remove:
	docker rmi -f `docker images | grep ${IMAGE_NAME} | tr -s ' ' | cut -d ' ' -f 3`
	docker rmi -f `docker images -f "dangling=true" -q`

docker-deploy:
	docker tag ${IMAGE_NAME}:${VERSION} ${FULL_IMAGE_NAME}
	docker push ${FULL_IMAGE_NAME}