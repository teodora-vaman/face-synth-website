import os
import random
from flask import Flask, render_template, request
import torch
import numpy as np
from icecream import ic
import base64
import io


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from static.Models.CGAN_64x64.discriminator_test import Discriminator
from static.Models.CGAN_64x64.generator_test import Generator

from static.Models.CGAN_128x128.discriminator_test import Discriminator as Discriminator_128
from static.Models.CGAN_128x128.generator import Generator as Generator_128

## MODELS ##
from static.Models.A2F_64x64.Discriminator_Stage2 import Discriminator as Discriminator_S2
from static.Models.A2F_64x64.Generator_Stage2 import Generator as Generator_S2
from static.Models.A2F_64x64.Discriminator_Stage3 import Discriminator as Discriminator_S3
from static.Models.A2F_64x64.Generator_Stage3 import Generator as Generator_S3
from static.Models.A2F_64x64.CVAE_Encoder import Encoder
from static.Models.A2F_64x64.CVAE_Decoder import Decoder

app = Flask(__name__)

@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/cgan64/')
def cgan64():
    # generated_image = generate_images()
    default_gender = 1

    return render_template('cgan64.html', default_gender=default_gender)

@app.route('/cgan128/')
def cgan128():
    # generated_image = generate_images()
    default_gender = 1

    return render_template('cgan128.html', default_gender=default_gender)


@app.route('/a2f/')
def a2f():
    # generated_image = generate_images()

    return render_template('a2f.html')


@app.route('/random_image/')
def randomImage():
    default_gender = 1

    return render_template('random_image.html', default_gender=default_gender)

@app.route('/button64', methods=['POST'])
def handle_button_click_cgan64():
    default_gender = "1"
    if request.method == 'POST':
        gender = int(request.form.get('gender'))
        
        generated_image = generate_imagesCGAN_64x64(attribut=gender)

        return render_template('cgan64.html', image_data=generated_image, default_gender=default_gender)
    return render_template('cgan64.html', default_gender=default_gender)

@app.route('/button128', methods=['POST'])
def handle_button_click_cgan128():
    default_gender = "1"
    attributes = [0, 1, 0, 0, 0, 0]
    if request.method == 'POST':
        gender = request.form.get('gender')
        age = request.form.get('age')
        glasses = request.form.get('glasses')
        bangs =  request.form.get('bangs')
        blackHair = request.form.get('blackHair')
        grayHair = request.form.get('grayHair')

        attributes = [0, 1, 0, 0, 0, 0]
        if gender is not None:
            attributes[0] = int(gender)
        if age is not None:
            attributes[1] = int(age)

        if glasses is not None:
            attributes[2] = int(glasses)

        if bangs is not None:
            attributes[3] = int(bangs)

        if blackHair is not None:
            attributes[4] = int(blackHair)

        if grayHair is not None:
            attributes[5] = int(grayHair)


        generated_image = generate_imagesCGAN_128(attribute=attributes)

        return render_template('cgan128.html', image_data=generated_image, male1=attributes[0], age1=attributes[1],
                               glasses1=attributes[2],bangs1=attributes[3],blackHair1=attributes[4],grayHair1=attributes[5])
    return render_template('cgan128.html', default_gender=default_gender)


@app.route('/buttonA2F', methods=['POST'])
def handle_button_click_a2f():
    default_gender = "1"
    attributes = [0, 0, 0, 1]
    if request.method == 'POST':
        gender = request.form.get('gender')
        smile = request.form.get('smile')
        glasses = request.form.get('glasses')
        beard =  request.form.get('beard')

        attributes = [0, 0, 0, 1]
        if gender is not None:
            attributes[0] = int(gender)

        if smile is not None:
            attributes[1] = int(smile)

        if glasses is not None:
            attributes[2] = int(glasses)

        if beard is not None:
            attributes[3] = int(beard)


        generated_image = synth_images(atribute=attributes)

        return render_template('a2f.html', image_data=generated_image )
    return render_template('a2f.html', default_gender=default_gender)




def generate_imagesCGAN_64x64(base_path='E:\Lucru\Dizertatie\Cod\Python_Website\static\Pretrained\CGAN_64x64\\15epoci', 
                    attribut=0, img_size=64):
# Dimensiunea vectorului latent
    dim_zgomot = 100

    retea_G = Generator(dim_zgomot=dim_zgomot, img_size=img_size)
    retea_G.load_state_dict(torch.load(base_path + '\\retea_Generator.pt'))

    retea_G.cuda()
    retea_G.eval()

    esantioane_proba = torch.randn(1, dim_zgomot, 1, 1)
    etichete_proba = torch.FloatTensor([[attribut]])

    etichete_proba = etichete_proba.to(torch.device('cuda'))
    esantioane_proba = esantioane_proba.to(torch.device('cuda'))
    imagini_generate = retea_G(esantioane_proba, etichete_proba).detach()

    imagini_generate = torch.squeeze(imagini_generate, 0)
    imagini_generate = imagini_generate.to(torch.device('cpu'))
    imagini_generate = np.transpose(imagini_generate,(0,1,2))
    # Convert the generated image tensor to a PIL image
    imagini_generate = transforms.ToPILImage()(imagini_generate.squeeze().cpu())

    # Encode the PIL image as a base64 string
    buffered = io.BytesIO()
    imagini_generate.save(buffered, format='PNG')
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image

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
    imagini_generate = transforms.ToPILImage()(imagini_generate.squeeze().cpu())

    # Encode the PIL image as a base64 string
    buffered = io.BytesIO()
    imagini_generate.save(buffered, format='PNG')
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image


def init_models(base_path, attribute_dim=4, img_size=64):
    encoder = Encoder(attribute_number=attribute_dim)
    decoder = Decoder(attribute_number=attribute_dim)
    retea_G3 = Generator_S3(img_size=3, attribute_number=attribute_dim)
    retea_D3 = Discriminator_S3(img_size=img_size, attribute_number=attribute_dim)
    retea_G2 = Generator_S2(attribute_number=attribute_dim)
    retea_D2 = Discriminator_S2(img_size=img_size, attribute_number=attribute_dim)
    retea_G3.cuda()
    retea_D3.cuda()
    retea_G2.cuda()
    retea_D2.cuda()
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()
    retea_G3.eval()
    retea_D3.eval()
    retea_G2.eval()
    retea_D2.eval()

    decoder.load_state_dict(torch.load(base_path + '\\retea_Decoder.pt'))
    encoder.load_state_dict(torch.load(base_path + '\\retea_Encoder.pt'))

    retea_D2.load_state_dict(torch.load(base_path + '\\retea_D_Stage2.pt'))
    retea_G2.load_state_dict(torch.load(base_path + '\\retea_G_Stage2.pt'))

    retea_D3.load_state_dict(torch.load(base_path + '\\A2F_retea_D_Stage3.pt'))
    retea_G3.load_state_dict(torch.load(base_path + '\\A2F_retea_G_Stage3.pt'))
    return encoder, decoder, retea_D2, retea_G2, retea_D3, retea_G3

base_path='E:\Lucru\Dizertatie\Cod\EvaluatingPerformance\Pretrained\A2F_64x64'
encoder, decoder, retea_D2, retea_G2, retea_D3, retea_G3 = init_models(base_path=base_path)


def synth_images(base_path='E:\Lucru\Dizertatie\Cod\Python_Website\static\Models\A2F_64x64', atribute=[0,0,0,1], img_size=64):
    # encoder, decoder, retea_D2, retea_G2, retea_D3, retea_G3 = init_models(base_path=base_path)
    DEVICE = 'cuda'

    sketch_img = torch.randn(3, img_size, img_size)
    esantioane_proba = torch.stack([sketch_img], dim=0)
    attributes = torch.FloatTensor([atribute])
        
    esantioane_proba = esantioane_proba.to(torch.device(DEVICE))
    attributes = attributes.to(torch.device(DEVICE))
    zgomot_proba = torch.FloatTensor(esantioane_proba.shape[0], 256).normal_(0, 1)
    zgomot_proba  = zgomot_proba.to(torch.device(DEVICE))

    zgomot_embedded, schita_embedded, encode_text = encoder(noise=zgomot_proba, attr_text=attributes, sketch=esantioane_proba, detach_flag=True)
    reconstructed_sketch_images, reconstructed_fake_images = decoder(zgomot_embedded[0], schita_embedded[0], detach_flag=True)

    imagini_generate_Stage2 = retea_G2(reconstructed_fake_images, encode_text).detach()
    imagini_generate_Stage3 = retea_G3(imagini_generate_Stage2, attributes).detach()

    imagini_generate_Stage3 = torch.squeeze(imagini_generate_Stage3, 0)
    imagini_generate_Stage3 = imagini_generate_Stage3.to(torch.device('cpu'))


    imagini_generate_Stage3 = np.transpose(imagini_generate_Stage3,(0,1,2))
    imagini_generate_Stage3 = transforms.ToPILImage()(imagini_generate_Stage3.squeeze().cpu())

    # Encode the PIL image as a base64 string
    buffered = io.BytesIO()
    imagini_generate_Stage3.save(buffered, format='PNG')
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image
    

if __name__ == '__main__':
   app.run(debug = True)

