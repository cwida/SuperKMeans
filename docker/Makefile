.PHONY: clean open-gpu

clean:
	docker image rm superkmeans-gpu 

build-gpu-image: Dockerfile.gpu
	@if [ -z "$(CPU_TARGET)" ]; then echo "CPU_TARGET is a required env variable, see GPU.md"; exit 2; \
	fi
	docker build --build-arg CPU_TARGET=$(CPU_TARGET) -t superkmeans-gpu -f Dockerfile.gpu ..

open-gpu-container:
	docker run -it --rm -w /home -v ..:/home --gpus all superkmeans-gpu /bin/bash
