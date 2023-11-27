# Imports
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Codificador
class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__()

        c_hid = base_channel_size

        self.net = nn.Sequential(

            # https://www.cs.toronto.edu/~kriz/cifar.html
            # 32x32 => 16x16
            nn.Conv2d(num_input_channels,
                      c_hid,
                      kernel_size=3,
                      padding=1,
                      stride=2),

            act_fn(),

            nn.Conv2d(c_hid,
                      c_hid,
                      kernel_size=3,
                      padding=1),

            act_fn(),

            # 16x16 => 8x8
            nn.Conv2d(c_hid,
                      2 * c_hid,
                      kernel_size=3,
                      padding=1,
                      stride=2),

            act_fn(),

            nn.Conv2d(2 * c_hid,
                      2 * c_hid,
                      kernel_size=3,
                      padding=1),

            act_fn(),

            # 8x8 => 4x4
            nn.Conv2d(2 * c_hid,
                      2 * c_hid,
                      kernel_size=3,
                      padding=1,
                      stride=2),

            act_fn(),

            nn.Flatten(),

            nn.Linear(2 * 16 * c_hid,
                      latent_dim)
        )

    def forward(self, x):
        return self.net(x)


# Decodificador
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__()

        c_hid = base_channel_size

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )

        self.net = nn.Sequential(

            # 4x4 => 8x8
            nn.ConvTranspose2d(2 * c_hid,
                               2 * c_hid,
                               kernel_size=3,
                               output_padding=1,
                               padding=1,
                               stride=2),

            act_fn(),

            nn.Conv2d(2 * c_hid,
                      2 * c_hid,
                      kernel_size=3,
                      padding=1),

            act_fn(),

            # 8x8 => 16x16
            nn.ConvTranspose2d(2 * c_hid,
                               c_hid,
                               kernel_size=3,
                               output_padding=1,
                               padding=1,
                               stride=2),

            act_fn(),

            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),

            act_fn(),

            # 16x16 => 32x32
            nn.ConvTranspose2d(c_hid,
                               num_input_channels,
                               kernel_size=3,
                               output_padding=1,
                               padding=1,
                               stride=2),

            # As imagens de entrada são dimensionadas entre -1 e 1, portanto, a saída também deve ser limitada
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


# DeepAutoencoder
class DeepAutoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()

        # Salvando os hiperparâmetros
        self.save_hyperparameters()

        # Criando encoder e decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

        # Exemplo de matriz de entrada necessária para visualizar o gráfico da rede
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


# Callback
class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstrói a imagem
            input_imgs = self.input_imgs.to(pl_module.device)

            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            # Plot e log
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    # seed
    pl.seed_everything(42)

    # Pasta de dados
    DATASET_PATH = "dados"

    # Pasta dos modelos
    CHECKPOINT_PATH = "modelos"

    # Transformações aplicadas em cada imagem
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Carregando o conjunto de dados de treinamento.
    imagens_treino = CIFAR10(root=DATASET_PATH,
                             train=True,
                             transform=transform,
                             download=True)

    # Divide os dados de treino em treino e validação
    dataset_treino, dataset_valid = torch.utils.data.random_split(imagens_treino, [45000, 5000])

    # Carregando o conjunto de dados de teste
    dataset_teste = CIFAR10(root=DATASET_PATH,
                            train=False,
                            transform=transform,
                            download=True)

    loader_treino = data.DataLoader(dataset_treino,
                                    batch_size=256,
                                    shuffle=True,
                                    drop_last=True,
                                    pin_memory=True,
                                    num_workers=4)

    loader_valid = data.DataLoader(dataset_valid,
                                   batch_size=256,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=4)

    loader_teste = data.DataLoader(dataset_teste,
                                   batch_size=256,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=4)


    # Função para obter uma imagem
    def get_train_images(num):
        return torch.stack([imagens_treino[i][0] for i in range(num)], dim=0)


    # Função para comparar imagens
    def compare_imgs(img1, img2, title_prefix=""):

        loss = F.mse_loss(img1, img2, reduction="sum")

        grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(4, 2))
        plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
        plt.imshow(grid)
        plt.axis('off')
        plt.show()


    for i in range(2):
        img, _ = imagens_treino[i]
        img_mean = img.mean(dim=[1, 2], keepdims=True)

        # Shift da imagem por 1 pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:, :1, :] = img_mean
        img_shifted[:, :, :1] = img_mean
        compare_imgs(img, img_shifted, "Shifted -")

        # Definir metade da imagem para zero
        img_masked = img.clone()
        img_masked[:, :img_masked.shape[1] // 2, :] = img_mean
        compare_imgs(img, img_masked, "Masked -")


    # Função de treinamento
    def treina_modelo(latent_dim):

        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
                             accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                             devices=1,
                             max_epochs=500,
                             callbacks=[ModelCheckpoint(save_weights_only=True),
                                        GenerateCallback(get_train_images(8), every_n_epochs=10),
                                        LearningRateMonitor("epoch")])

        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None

        pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")

        # Se encontra o modelo já treinado, ele será usado. Caso contrário, será feito o treinamento.
        if os.path.isfile(pretrained_filename):
            print("Encontrei o modelo treinado. Carregando...")
            model = DeepAutoencoder.load_from_checkpoint(pretrained_filename)
        else:
            print("Iniciando o treinamento...")
            model = DeepAutoencoder(base_channel_size=32, latent_dim=latent_dim)
            trainer.fit(model, loader_treino, loader_valid)

        val_result = trainer.test(model, loader_valid, verbose=False)
        test_result = trainer.test(model, loader_teste, verbose=False)

        result = {"teste": test_result, "valid": val_result}

        return model, result

    ''' Comparando a Dimensionalidade Latente '''

    # Nível de precisão das operações de multiplicação
    torch.set_float32_matmul_precision('high')

    # Dicionário
    model_dict = {}

    # Loop
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = treina_modelo(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]["result"]["valid"][0]["test_loss"] for k in latent_dims]

    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.plot(latent_dims,
             val_scores, '--',
             color="#000",
             marker="*",
             markeredgecolor="#000",
             markerfacecolor="y",
             markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Erro de reconstrução sobre dimensionalidade latente", fontsize=14)
    plt.xlabel("Dimensionalidade latente")
    plt.ylabel("Erro de reconstrução")
    plt.minorticks_off()
    plt.ylim(0, 100)
    plt.show()


    # Função de visualização
    def visualize_reconstructions(model, input_imgs):

        model.eval()

        with torch.no_grad():
            reconst_imgs = model(input_imgs.to(model.device))

        reconst_imgs = reconst_imgs.cpu()

        # Plot
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(7, 4.5))
        plt.title(f"Reconstruído a partir de {model.hparams.latent_dim} latentes")
        plt.imshow(grid)
        plt.axis('off')
        plt.show()


    input_imgs = get_train_images(4)

    # Loop
    for latent_dim in model_dict:
        visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs)

    ''' Explorando Limitações do Modelo '''

    rand_imgs = torch.rand(2, 3, 32, 32) * 2 - 1
    visualize_reconstructions(model_dict[256]["model"], rand_imgs)

    plain_imgs = torch.zeros(4, 3, 32, 32)

    # Canal de cor única
    plain_imgs[1, 0] = 1

    # Checkboard
    plain_imgs[2, :, :16, :16] = 1
    plain_imgs[2, :, 16:, 16:] = -1

    # Progressão de cores
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32))
    plain_imgs[3, 0, :, :] = xx
    plain_imgs[3, 1, :, :] = yy

    visualize_reconstructions(model_dict[256]["model"], plain_imgs)

    ''' Gerando Novas Imagens '''

    model = model_dict[256]["model"]

    latent_vectors = torch.randn(8, model.hparams.latent_dim, device=model.device)

    with torch.no_grad():
        imgs = model.decoder(latent_vectors)
        imgs = imgs.cpu()

    # Grid
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.5)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(8, 5))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

    ''' Encontrando Imagens Visualmente Semelhantes '''

    # Se você quiser tentar uma dimensionalidade latente diferente, mude aqui!
    model = model_dict[128]["model"]


    def embed_imgs(model, data_loader):

        # Encode de todas as imagens no data_laoder usando o modelo e retorna imagens e codificações
        img_list, embed_list = [], []
        model.eval()
        for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
            with torch.no_grad():
                z = model.encoder(imgs.to(model.device))

            img_list.append(imgs)

            embed_list.append(z)

        return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))


    train_img_embeds = embed_imgs(model, loader_treino)
    test_img_embeds = embed_imgs(model, loader_teste)


    # Função do sistema de similaridade
    def find_similar_images(query_img, query_z, key_embeds, K=8):
        dist = torch.cdist(query_z[None, :], key_embeds[1], p=2)
        dist = dist.squeeze(dim=0)
        dist, indices = torch.sort(dist)
        imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_display, nrow=K + 1, normalize=True)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(12, 3))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()


    # Plot das imagens mais próximas para as primeiras N imagens de teste como exemplo
    for i in range(8):
        find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)

