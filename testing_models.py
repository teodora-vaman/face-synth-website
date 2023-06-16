import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from static.Models.CGAN_64x64.discriminator_test import Discriminator
from static.Models.CGAN_64x64.generator_test import Generator

from static.Models.CGAN_128x128.discriminator_test import Discriminator as Discriminator_128
from static.Models.CGAN_128x128.generator_test import Generator as Generator_128

def generate_imagesCGAN_128(base_path = 'E:\Lucru\Dizertatie\Cod\Python_Website\static\Pretrained\CGAN_128x128',img_size=128,attribute=[0,1,0,0,0,0], display = 0):
    dim_zgomot = 100

    retea_G = Generator_128(dim_zgomot=dim_zgomot, img_size=img_size, attribute_number=6)

    retea_G.load_state_dict(torch.load(base_path + '\\retea_Generator.pt'))
    retea_G.cuda()
    retea_G.eval()

    esantioane_proba = torch.randn(1, dim_zgomot, 1, 1)
    etichete_proba = torch.FloatTensor([attribute])

    etichete_proba = etichete_proba.to(torch.device('cuda'))
    esantioane_proba = esantioane_proba.to(torch.device('cuda'))

    imagini_generate = retea_G(esantioane_proba, etichete_proba).detach()
    imagini_generate = torch.squeeze(imagini_generate, 0)
    imagini_generate = imagini_generate.to(torch.device('cpu'))
    imagini_generate = np.transpose(imagini_generate,(0,1,2))
    # imagini_generate = transforms.ToPILImage()(imagini_generate.squeeze().cpu())

    return imagini_generate


imagini_generate = generate_imagesCGAN_128()
plt.figure()
plt.imshow(np.transpose(imagini_generate,(2,1,0)))
plt.show()