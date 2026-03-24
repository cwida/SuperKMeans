.PHONY: clean docker-reset open-cpu

clean:
	docker image rm kmeans-cpu 
	docker image rm kmeans-gpu 

# For some reason this is sometimes necessary
docker-reset:
	systemctl --user daemon-reload
	systemctl --user restart docker

build-cpu: Dockerfile.cpu
	docker build -t kmeans-cpu -f Dockerfile.cpu ..

open-cpu:
	docker run -it --rm -v ..:/home kmeans-cpu /bin/bash

build-gpu: Dockerfile.gpu
	docker build -t kmeans-gpu -f Dockerfile.gpu ..

open-gpu:
	docker run -it --rm -v ..:/home -v /mnt/scratch/shepkema/vector-data/processed:/data --gpus all kmeans-gpu /bin/bash
