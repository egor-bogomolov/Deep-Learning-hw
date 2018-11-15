import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.vae import VAE, loss_function
from vae.trainer import VAETrainer


def get_config():
    parser = argparse.ArgumentParser(description='Training VAE on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--starting-epoch', type=int, default=0,
                        help='first epoch')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--imИ какage-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    parser.add_argument('--pretrained-vae', type=str, help='path to pretrained net')
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = datasets.CIFAR10(root=config.data_root, download=True,
                                     transform=transform, train=True)
    dataset_test = datasets.CIFAR10(root=config.data_root, download=True,
                                    transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    vae = VAE(config.image_size)

    if config.pretrained_vae:
        vae.load_state_dict(torch.load(config.pretrained_vae))

    trainer = VAETrainer(model=vae,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         optimizer=Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                         loss_function=loss_function,
                         device='cuda')

    for epoch in range(config.starting_epoch, config.starting_epoch + config.epochs):
        trainer.train(epoch, config.log_metrics_every)
        trainer.test(epoch, config.log_metrics_every, config.show_img_every, config.n_show_samples)


if __name__ == '__main__':
    main()
