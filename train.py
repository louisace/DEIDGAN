import torch.backends.cudnn as cudnn
import torch.optim as optim
from Models.models import MultiscaleDiscriminator, GANLoss, MultiscaleDiscriminator2, Generator, VGGLoss
import itertools
from data_train import Data_train
from parameter import *
from Models.utils import *
from tensorboardX import SummaryWriter
from torchvision.utils import save_image, make_grid
import warnings
from torch import autograd
from tqdm import tqdm
import os
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class Train():

    def __init__(self, config):
        self.config = config
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.max_epoch = config.max_epoch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        
        self.lr_update_step = config.lr_update_step
        self.n_layers_D = config.n_layers_D
        self.num_D = config.num_D
        self.gan_mode = config.gan_mode
        self.FloatTensor = torch.cuda.FloatTensor
        self.semantic_nc = config.semantic_nc
        # Data
        self.datasets = Data_train(self.batch_size, self.image_size)

        # logs path
        self.log_path = config.log_path
        self.tensorboard_path = config.tensorboard_path
        self.model_save_path = config.model_save_path
        self.samples_path = config.samples_path
        make_folder(self.log_path)
        make_folder(self.tensorboard_path)
        make_folder(self.model_save_path)
        make_folder(self.samples_path)

    def build_model(self):
        print('=====>loading SPADE  model')
        self.netG = Generator().cuda()
        self.netD = MultiscaleDiscriminator().cuda()
        self.netD2 = MultiscaleDiscriminator2().cuda()
        self.netD3 = MultiscaleDiscriminator2().cuda()
        self.netG = torch.nn.DataParallel(self.netG)
        self.netD = torch.nn.DataParallel(self.netD)
        self.netD2 = torch.nn.DataParallel(self.netD2)
        self.netD3 = torch.nn.DataParallel(self.netD3)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(self.gan_mode, tensor=self.FloatTensor)
        self.criterionVGG = VGGLoss()

    def train(self):
        self.build_model()
        dataloader = torch.utils.data.DataLoader(dataset=self.datasets, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

        optimizer_g = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_d = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_d2 = optim.Adam(itertools.chain(self.netD2.parameters(), self.netD3.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
        g_lr = self.lr
        d_lr = self.lr

        writer = SummaryWriter(self.tensorboard_path)
        for iter in range(1, self.max_epoch + 1):
            for idx, data in tqdm(enumerate(dataloader, 0)):
                image, image_bg, labels, labels_ori, image11, image22 = data[0], data[1], data[2], data[3], data[4], data[5]

                size = labels.size()
                oneHot_size = (size[0], self.semantic_nc, size[2], size[3])
                labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_mask_org = labels_real.scatter_(1, labels_ori.data.long().cuda(), 1.0)

                oneHot_size = (size[0], self.semantic_nc, size[2], size[3])
                labels_ref = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_mask_ref = labels_ref.scatter_(1, labels.data.long().cuda(), 1.0)

                image = image.cuda()
                image_bg = image_bg.cuda()
                image11 = image11.cuda()
                image22 = image22.cuda()

                # ------------ update discriminator----------------
                requires_grad(self.netG, False)
                requires_grad(self.netD, True)
                requires_grad(self.netD2, False)
                requires_grad(self.netD3, False)

                optimizer_d.zero_grad()
                image.requires_grad_()
                pred_real = self.netD(torch.cat([image, input_mask_org], 1))
                loss_d_real = self.criterionGAN(pred_real, True)
                loss_d_real.backward()

                with torch.no_grad():
                    fake_image = self.netG(input_mask_org, image, image_bg)

                pred_fake = self.netD(torch.cat([fake_image.detach(), input_mask_org], 1))
                loss_d_fake = self.criterionGAN(pred_fake, False)
                loss_d_fake.backward()
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                optimizer_d.step()

                requires_grad(self.netG, False)
                requires_grad(self.netD, False)
                requires_grad(self.netD2, True)
                requires_grad(self.netD3, True)
                optimizer_d2.zero_grad()
                pred_real_Deid = self.netD2(torch.cat([image, image11], 1))
                pred_real_IDS = self.netD3(torch.cat([image, image11], 1))
                loss_d_real_Deid = self.criterionGAN(pred_real_Deid, True)
                loss_d_real_IDS = self.criterionGAN(pred_real_IDS, True)
                loss_d_real2 = loss_d_real_Deid + loss_d_real_IDS

                pred_fake_Deid = self.netD2(torch.cat([fake_image.detach(), image11], 1))
                pred_fake_IDS = self.netD3(torch.cat([fake_image.detach(), image22], 1))
                loss_d_fake_Deid = self.criterionGAN(pred_fake_Deid, False)
                loss_d_fake_IDS = self.criterionGAN(pred_fake_IDS, False)
                loss_d_fake2 = loss_d_fake_Deid + loss_d_fake_IDS
                loss_d2 = (loss_d_real2 + loss_d_fake2) * 0.5
                loss_d2.backward()
                optimizer_d2.step()

                # ------------ update generator----------------
                requires_grad(self.netG, True)
                requires_grad(self.netD, False)
                requires_grad(self.netD2, False)
                requires_grad(self.netD3, False)
                optimizer_g.zero_grad()
                fake_image = self.netG(input_mask_org, image, image_bg)
                pred_image = self.netD(torch.cat([fake_image, input_mask_org], 1))
                pred_image_Deid = self.netD2(torch.cat([fake_image, image11], 1))
                pred_image_IDS = self.netD3(torch.cat([fake_image, image22], 1))
                loss_g_gan = self.criterionGAN(pred_image, True)
                loss_g_gan_Deid = self.criterionGAN(pred_image_Deid, False)
                loss_g_gan_IDS = self.criterionGAN(pred_image_IDS, False) * 0.0001
                loss_g_gan2 = loss_g_gan_Deid + loss_g_gan_IDS

                # style loss
                # loss_g_vgg_style = 0
                loss_g_vgg_style = self.criterionVGG(image.detach(), fake_image) * 0.1

                loss_g_feat = 0
                pred_fake = self.netD(torch.cat([fake_image, input_mask_org], 1))
                pred_real = self.netD(torch.cat([image, input_mask_org], 1))
                num_D = len(pred_fake)
                for i in range(num_D):
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):
                        unweighted_loss = self.criterionL1(pred_fake[i][j], pred_real[i][j].detach())
                        loss_g_feat += unweighted_loss
                loss_g = loss_g_gan + loss_g_feat + loss_g_gan2

                loss_g.backward()
                optimizer_g.step()

                print("===> iter({}/{})({}/{}): loss_d: {:.4f} loss_g_gan: {:.4f} loss_g_vgg_style: {:.4f} loss_g_feat: {:.4f}".format(iter, self.max_epoch, idx, len(dataloader), loss_d.item(), loss_g_gan.item(), loss_g_vgg_style.item(), loss_g_feat))

                writer.add_scalar('loss_d1/d_real', loss_d_real.item(), global_step=len(dataloader)*(iter-1)+idx)
                writer.add_scalar('loss_d1/d_fake', loss_d_fake.item(), global_step=len(dataloader)*(iter-1)+idx)
                writer.add_scalar('loss_g/g_gan', loss_g_gan.item(), global_step=len(dataloader)*(iter-1)+idx)
                writer.add_scalar('loss_g/g_vgg_style', loss_g_vgg_style.item(), global_step=len(dataloader) * (iter - 1) + idx)
                writer.add_scalar('loss_g/gan_feat', loss_g_feat, global_step=len(dataloader) * (iter-1) + idx)
                writer.add_scalar('loss_d2/loss_d_real_Deid', loss_d_real_Deid.item(),
                                  global_step=len(dataloader)*(iter-1)+idx)
                writer.add_scalar('loss_d2/loss_d_real_IDS', loss_d_real_IDS.item(),
                                  global_step=len(dataloader) * (iter - 1) + idx)
                writer.add_scalar('loss_d2/loss_d_fake_Deid', loss_d_fake_Deid.item(),
                                  global_step=len(dataloader) * (iter - 1) + idx)
                writer.add_scalar('loss_d2/loss_d_fake_IDS', loss_d_fake_IDS.item(),
                                  global_step=len(dataloader)*(iter-1)+idx)
                writer.add_scalar('loss_g/loss_g_gan_Deid', loss_g_gan_Deid,
                                  global_step=len(dataloader) * (iter - 1) + idx)
                writer.add_scalar('loss_g/loss_g_gan_IDS', loss_g_gan_IDS,
                                  global_step=len(dataloader) * (iter - 1) + idx)

                mask_colors_org = generate_label_color(input_mask_org, self.image_size, self.semantic_nc)
                mask_colors_ref = generate_label_color(input_mask_ref, self.image_size, self.semantic_nc)

                tensor_lsit = [image, fake_image, mask_colors_org, mask_colors_ref]
                visual(tensor_lsit, writer, len(dataloader)*(iter-1)+idx)
                
                image_com = mask_colors_org.detach().float()
                image_com = torch.cat([image_com, image.detach().cpu()], 0)
                image_com = torch.cat([image_com, mask_colors_ref.float().detach().cpu()], 0)
                image_com = torch.cat([image_com, fake_image.detach().cpu()], 0)
                if (len(dataloader)*(iter-1)+idx) % 1000 == 0:
                    save_image(image_com.data, os.path.join(self.samples_path, 'real_samples_%06d.jpg'% (len(dataloader)*(iter-1)+idx)), nrow=self.batch_size, normalize=True)

            # checkpoint
            if iter >= 30 and iter % 5 == 0:
                netG_model_out_path = os.path.join(self.model_save_path, 'netG_epoch_%d.pth' % (iter))
                netD_model_out_path = os.path.join(self.model_save_path, 'netD_epoch_%d.pth' % (iter))
                netD2_model_out_path = os.path.join(self.model_save_path, 'netD2_epoch_%d.pth' % (iter))
                netD3_model_out_path = os.path.join(self.model_save_path, 'netD3_epoch_%d.pth' % (iter))
                netG_state = {
                    'state': self.netG.module.state_dict() if hasattr(self.netG, 'module') else self.netG.state_dict(),
                    'epoch': iter,
                    'lr': g_lr
                }
                netD_state = {
                    'state': self.netD.module.state_dict() if hasattr(self.netD, 'module') else self.netD.state_dict(),
                    'epoch': iter,
                }
                netD2_state = {
                    'state': self.netD2.module.state_dict() if hasattr(self.netD2, 'module') else self.netD2.state_dict(),
                    'epoch': iter,
                }
                netD3_state = {
                    'state': self.netD3.module.state_dict() if hasattr(self.netD3, 'module') else self.netD3.state_dict(),
                    'epoch': iter,
                }
                torch.save(netG_state, netG_model_out_path)
                torch.save(netD_state, netD_model_out_path)
                torch.save(netD2_state, netD2_model_out_path)
                torch.save(netD3_state, netD3_model_out_path)
                print("Checkpoint saved to %s" % (self.model_save_path))

            if iter % self.lr_update_step == 0:
                g_lr = g_lr * 0.5
                d_lr = d_lr * 0.5
                self.update_lr(optimizer_g, optimizer_d, g_lr, d_lr)
                print('Decayed learning rates, g_lr: {} d_lr: {}.'.format(g_lr, d_lr))


    def update_lr(self, optimizer_g, optimizer_d, g_lr, d_lr):
        for param_group in optimizer_g.param_groups:
            param_group['lr'] = g_lr
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = d_lr

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def main(config):
    cudnn.benchmark = True

    trainer = Train(config)
    trainer.train()


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)

