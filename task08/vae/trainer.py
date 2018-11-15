import logging
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class VAETrainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, image_size, device='cpu'):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.image_size = image_size
        self.device = device
        self.model = model.to(device)
        self.writer = SummaryWriter()

    def train(self, epoch, log_interval):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            self.model.zero_grad()

            data = data.to(self.device)
            decoded, mu, logvar = self.model(data)
            train_loss = self.loss_function(decoded, data, mu, logvar)
            train_loss.backward()

            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            self.optimizer.step()

            if batch_idx % log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    norm_train_loss)
                logging.info(msg)

                batch_size = self.train_loader.batch_size
                train_size = len(self.train_loader.dataset)
                batches_per_epoch_train = train_size // batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)

    def test(self, epoch, log_interval, image_log_interval, n_show_samples):
        self.model.eval()
        test_epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):

            with torch.no_grad():
                data = data.to(self.device)
                decoded, mu, logvar = self.model(data)
                decoded = decoded.view(-1, 3, self.image_size, self.image_size)
                test_loss = self.loss_function(decoded, data, mu, logvar)

            test_epoch_loss += test_loss

            batch_size = self.test_loader.batch_size
            batches_per_epoch_test = len(self.test_loader.dataset) // batch_size
            global_step = batches_per_epoch_test * epoch + batch_idx
            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    test_loss / len(data))
                logging.info(msg)


                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=global_step)

            if batch_idx % image_log_interval == 0:
                self.plot_generated(decoded, data, n_show_samples, global_step)

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)

    def plot_generated(self, decoded, real, n_show_samples, global_step):
        x = vutils.make_grid(decoded[:n_show_samples], normalize=True, scale_each=True)
        self.writer.add_image('img/decoded', x, global_step)

        y = vutils.make_grid(real[:n_show_samples], normalize=True, scale_each=True)
        self.writer.add_image('img/real', y, global_step)
        pass

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
