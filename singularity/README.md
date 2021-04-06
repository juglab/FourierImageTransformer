# Build Singularity Container

This is the procedure we used to build singularity containers. The singularity recipes were generated
with [neurodocker](https://github.com/ReproNim/neurodocker).

Build Python package:
`python setup.py bdist_wheel`

Build singularity recipe:
`neurodocker generate singularity -b nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 -p apt --copy ./dist/fourier_image_transformer-0.2.0-py3-none-any.whl /fourier_image_transformer-0.2.0-py3-none-any.whl --miniconda create_env=fit conda_install='python=3.6 astra-toolbox pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c astra-toolbox/label/dev' pip_install='/fourier_image_transformer-0.2.0-py3-none-any.whl' activate=true --entrypoint "/neurodocker/startup.sh python" > singularity/v0.2.0.Singularity`

Build singularity container:
`sudo singularity build singularity/fit_v0.2.0.simg singularity/v0.2.0.Singularity`
