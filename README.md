# Fourier Image Transformer

Tim-Oliver Buchholz<sup>1</sup> and Florian Jug<sup>2</sup></br>
<sup>1</sup>tibuch@mpi-cbg.de, <sup>2</sup>florian.jug@fht.org

Transformer architectures show spectacular performance on NLP tasks and have recently also been used for tasks such as
image completion or image classification. Here we propose to use a sequential image representation, where each prefix of
the complete sequence describes the whole image at reduced resolution. Using such Fourier Domain Encodings (FDEs), an
auto-regressive image completion task is equivalent to predicting a higher resolution output given a low-resolution
input. Additionally, we show that an encoder-decoder setup can be used to query arbitrary Fourier coefficients given a
set of Fourier domain observations. We demonstrate the practicality of this approach in the context of computed
tomography (CT) image reconstruction. In summary, we show that Fourier Image Transformer (FIT) can be used to solve
relevant image analysis tasks in Fourier space, a domain inherently inaccessible to convolutional architectures.

Preprint: [arXiv](arXiv)

## FIT for Super-Resolution

![SRes](figs/SRes.png)

__FIT for super-resolution.__ Low-resolution input images are first transformed into Fourier space and then unrolled into an
FDE sequence, as described in Section 3.1 of the paper. This FDE sequence can now be fed to a FIT, that, conditioned on this input,
extends the FDE sequence to represent a higher resolution image. This setup is trained using an FC-Loss that enforces
consistency between predicted and ground truth Fourier coefficients. During inference, the FIT is conditioned on the
first 39 entries of the FDE, corresponding to (a,d) 3x Fourier binned input images. Panels (b,e) show the inverse Fourier
transform of the predicted output, and panels (c,f) depict the corresponding ground truth.

## FIT for Tomography

![TRec](figs/TRec.png)

__FIT for computed tomography.__ We propose an encoder-decoder based Fourier Image Transformer setup for tomographic
reconstruction. In 2D computed tomography, 1D projections of an imaged sample (i.e. the columns of a sinogram) are
back-transformed into a 2D image. A common method for this transformationis the filtered backprojection (FBP). Since
each projection maps to a line of coefficients in 2D Fourier space, a limited number of projections in a sinogram leads
to visible streaking artefacts due to missing/unobserved Fourier coefficients. The idea of our FIT setup is to encode
all information of a given sinogram and use the decoder to predict missing Fourier coefficients. The reconstructed image
is then computed via an inverse Fourier transform (iFFT) of these predictions. In order to reduce high frequency
fluctuations in this result, we introduce a shallow conv-block after the iFFT (shown in black). We train this setup
combining the FC-Loss, see Section 3.2 in the paper, and a conventional MSE-loss between prediction and ground truth.

## Installation TODO

astra-toolbox requires cuda 10.2: conda install -c astra-toolbox/label/dev astra-toolbox

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

Build Python package:
`python setup.py bdist_wheel`

Build singularity recipe:
`neurodocker generate singularity -b nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 -p apt --copy /home/tibuch/Gitrepos/FourierImageTransformer/dist/fourier_image_transformers-0.1.24_zero-py3-none-any.whl /fourier_image_transformers-0.1.24_zero-py3-none-any.whl --miniconda create_env=fit conda_install='python=3.7 astra-toolbox pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c astra-toolbox/label/dev' pip_install='/fourier_image_transformers-0.1.24_zero-py3-none-any.whl' activate=true --entrypoint "/neurodocker/startup.sh python" > v0.1.24_zero.Singularity`

Build singularity container:
`sudo singularity build fit_v0.1.24_zero.simg v0.1.24_zero.Singularity`
