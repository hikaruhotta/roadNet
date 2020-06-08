<img src="https://github.com/HikaruHotta/roadNet/blob/master/images/outputExample.png" height="300" />

# CS231n Road Segmentation Project
The interconnected society we live in is highly dependent on roads for transportation. However, obtaining a good ground truth mapping of road locations may require significant location data, or mapping roads manually, which is an enormous task. This is especially problematic in rural areas and less developed areas. 

In this project, we investigated automatic road network generation using satellite imagery. Our aim was to develop a deep neural network that is robust enough to generate segmentations of roads from both urban and rural satellite images. 

For a more detailed explanation, refer to our paper in the main repository, labelled "CS_231N_Paper.pdf".

**Dataset** *(MassDataManipulation.ipynb)*

We used the Massachusetts Roads Dataset, developed as part of a PhD thesis; it contains images primarily from Massachusetts.
The original dataset contains 151 large images, over 2600 sq km, of urban to rural regions. We cropped images to 256 x 256 for a more manageable for model. We also normalized with channel-wise mean and standard deviation. The result images were 18000+ training, 350 valid, 1200+ test. 

**Creating the UNet** *(UNetBaseline.ipynb)*

To conduct semantic road segmentation, we trained a UNet model on the Massachusetts Roads Dataset. This architecture gets its name from it;s U-shaped decoder and encoder architecture. The skip connections between the decoding and encoding layers allow the network to propagate contextual information to higher resolution layers. We trained our model for 40 epochs, which produced outputs that capture the structure of most roads. However, they were generally noisy and while thresholding reduced noise significantly, it compromised road connectivity in our maps.
We referenced: https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

**Pix2Pix cGAN** *(roadPix2PixcGAN.ipynb)*

Therefore, we proposed a Pix2Pix cGAN to take a deeplearning based approach to denoising. We used a generator architecture identical to our UNet Model with the only difference being the input has one channel instead of three. For our discriminator, we decided to use a patchGAN architecture. This is the training arc for our cGAN where the generator tries to minimize the adversarial objective while the discriminator tries to maximize it.

