# DDPM-for-De-Novo-Protein-Sequence-Generation
In this repo, we implement and train a Denoising Diffusion Probabilistic Model (DDPM) to generate De Novo protein backbone structures.

## Build image
docker build -t protein-diffusion .

## Run container
docker run --rm --gpus all --env PYTHONPATH=":/app/src" --volume .:/app -it protein-diffusion
