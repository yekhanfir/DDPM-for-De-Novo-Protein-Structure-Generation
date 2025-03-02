import torch
from torch import nn
from model.model_utils import get_protein

class DDPM(nn.Module):
    """
    DDPM paper by Ho. et al: https://arxiv.org/abs/2006.11239.
    Complete implementation by Umar Jamil: https://github.com/hkproj/pytorch-ddpm/.

    A minimal implementation of the DDPM framework,
    using the UNet architecture defined above.

    self.timesteps: maximum noising / denoising timesteps.

    self.betas: linear noise schedule, used to decide the amount
    that should be added of the total sampled Gaussian noise.

    self.alphas: used to compute the cumulative product of the transformer
    noise schedule, allowing to obtain noisy input at an arbitrary timestep t
    in one step.

    self.alphas_cumprod: cumulative products of the alphas.
    """
    def __init__(self, unet, device, model_config, max_seq_len):
        super(DDPM, self).__init__()
        
        self.device = device
        self.unet = unet
        self.max_seq_len = max_seq_len
        self.timesteps = model_config.timesteps
        self.betas = torch.linspace(
            model_config.beta_start, 
            model_config.beta_end, 
            model_config.timesteps
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t, noise):
        """
        applies noise to an input x0 to reach xt, where t
        an arbitrary timestem.
        """
        t_on_cpu = t.detach().cpu()

        sqrt_alpha_cumprod = torch.sqrt(
            self.alphas_cumprod[t_on_cpu]
        ).view(-1,1,1,1).to(self.device)

        sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1 - self.alphas_cumprod[t_on_cpu]
        ).view(-1,1,1,1).to(self.device)

        noisy_x = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return noisy_x

    def denoise_at_t(self, noisy_x, predicted_noise, t):
        """
        Removes given predicted noise at timestep t from the noisy input,
        and returns the denoised structure.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])

        denoised_struct = (1/sqrt_alpha_cumprod) * (
            noisy_x - (self.betas[t] / sqrt_one_minus_alpha_cumprod) * predicted_noise
        )
        return denoised_struct

    def get_predicted_protein(self, noisy_x, predicted_noise, t):
        """
        Given a noisy input (noisy_x), predicted_noise and timestep (t),
        Removes the predicted noise from the noisy input, and returns
        a Protein instance of the denoised structure.
        """
        predicted_struct = self.denoise_at_t(noisy_x, predicted_noise, t)
        prot = get_protein(predicted_struct, self.max_seq_length)
        return prot


    def sample(self, shape, device):
        """
        Starts by sampling pure Gaussian noise,
        then iteratively denoising it at each timestep.
        Also adds an additional amount of noise (amount depends on t)
        to insure stochasticity and diversity.
        """
        predicted_noise_over_T = []
        with torch.no_grad():
            x = torch.randn(shape, device=device)
            denoised_over_T = [x]
            for t in reversed(range(self.timesteps)):
                z = torch.randn(shape, device=device) if t > 0 else 0
                tmp_t = torch.tensor(t, device=device).view(-1, 1)
                predicted_noise = self.unet(x, tmp_t.float())
                predicted_noise_over_T.append(predicted_noise)
                x = self.denoise_at_t(x, predicted_noise, t) + torch.sqrt(self.betas[t])*z
                denoised_over_T.append(x)
        return x, denoised_over_T, predicted_noise_over_T