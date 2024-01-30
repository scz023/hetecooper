import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, recon_x_a, x_a, recon_x_b, x_b, latent_a, latent_b):
        # 重建损失
        recon_loss_a = F.mse_loss(recon_x_a, x_a)
        recon_loss_b = F.mse_loss(recon_x_b, x_b)

        # 潜在变量的差异损失
        latent_loss = F.mse_loss(latent_a, latent_b)

        # 交叉重建损失
        cross_recon_loss_a = F.mse_loss(recon_x_a, x_b)
        cross_recon_loss_b = F.mse_loss(recon_x_b, x_a)

        # 总损失
        total_loss = self.alpha * (recon_loss_a + recon_loss_b) + \
                     self.beta * latent_loss + \
                     self.gamma * (cross_recon_loss_a + cross_recon_loss_b)
        return total_loss

    def logging(self, epoch, batch_id, batch_len, total_loss, writer=None, pbar=None):
        print_msg = "[epoch {}][{}/{}], || Total Loss: {:.2f}".format(epoch, batch_id + 1, batch_len, total_loss)   
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)