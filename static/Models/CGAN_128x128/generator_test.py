import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic

class Generator(nn.Module):
    def __init__(self, dim_zgomot, img_size, attribute_number = 1):
        super(Generator, self).__init__()
        self.img_size = img_size
        embeding_size = 50
        self.embedd = nn.Embedding(attribute_number, embeding_size)

        self.embedding_attribute =  nn.Sequential(
            nn.Linear(in_features=attribute_number, out_features=embeding_size, bias=False), # batch_nr x nr_attribute => batch x 1 out: batch x 256
            nn.BatchNorm1d(embeding_size),
            nn.ReLU(True),
        )

        # intrare vector latent - nr_imag x 100 x 1 x 1
        self.tconv1 = nn.ConvTranspose2d(in_channels=dim_zgomot + embeding_size, 
                                         out_channels=128, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        # nr_imax x 128 x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(in_channels=128, 
                                         out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        # nr_imag x 64 x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(in_channels=64, 
                                         out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        # nr_imag x 32 x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(in_channels=32, 
                                         out_channels=16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(True)

        # nr_imag x 16 x 32 x 32
        self.tconv5 = nn.ConvTranspose2d(in_channels=16, 
                                         out_channels=8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU(True)

        # nr_imag x 8 x 64 x 64
        self.tconv6 = nn.ConvTranspose2d(in_channels=8, 
                                         out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

        # nr_imag x 3 x 128 x 128
        self.out = nn.Sigmoid()

    
    def forward(self, input, labels):
        
        # print(labels.shape)
        embedded_labels = self.embedding_attribute(labels.float()).unsqueeze(2).unsqueeze(3)
        # print(input.shape)
        # print(embedded_labels.shape)
        x = torch.cat([input, embedded_labels], dim=1)

        x = self.tconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.tconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.tconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.tconv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.tconv6(x)

        return self.out(x)


if __name__ == "__main__":
    image_size = [1,128,128] # input img: 64 x 64 for CelebA
    x = torch.randn(4, 3, 128, 128)
            
    vector_generare = torch.randn(len(x), 100, 1, 1)

    y = torch.LongTensor([0])  
    image_size = [1,128,128] # input img: 64 x 64 for CelebA
    test_y = torch.LongTensor([[0], [0], [1], [1]])

    retea_G = Generator(dim_zgomot = 100, img_size=128, attribute_number=1)
    result = retea_G(vector_generare, test_y)
    ic(result.shape)  # trebuie sa fie 3 x 64 x 64
