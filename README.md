astra-toolbox requires cuda 10.2: conda install -c astra-toolbox/label/dev astra-toolbox

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

Build Python package:
`python setup.py bdist_wheel`

Build singularity recipe:
`neurodocker generate singularity -b nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 -p apt --copy /home/tibuch/Gitrepos/FourierImageTransformer/dist/fourier_image_transformers-0.1.8-py3-none-any.whl /fourier_image_transformers-0.1.8-py3-none-any.whl --miniconda create_env=fit conda_install='python=3.7 astra-toolbox pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c astra-toolbox/label/dev' pip_install='/fourier_image_transformers-0.1.8-py3-none-any.whl' activate=true --entrypoint "/neurodocker/startup.sh python" > v0.1.8.Singularity`

Build singularity container:
`sudo singularity build fit_v0.1.8.simg v0.1.8.Singularity`