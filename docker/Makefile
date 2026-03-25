.PHONY: clean open-gpu

clean:
	docker image rm superkmeans-gpu 

build-gpu-image: Dockerfile.gpu
	docker build -t superkmeans-gpu -f Dockerfile.gpu ..

open-gpu-container:
	docker run -it --rm -v ..:/home --gpus all superkmeans-gpu /bin/bash
