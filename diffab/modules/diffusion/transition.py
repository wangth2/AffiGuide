import numpy as np
import torch
import torch.nn as nn

from diffab.modules.common.layers import clampped_one_hot
from diffab.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec
from diffab.tools.eval.hydropathy import hydropathy_prob


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        """
        Args:
            p_0: Positions in normalized diffusion scale, (N, L, 3).
            mask_generate: Generated residue mask, (N, L).
            t: Time steps, (N,).
        Returns:
            p_noisy: Noisy positions at step t, (N, L, 3).
            e_rand: Added Gaussian noise, (N, L, 3).
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0 * p_0 + c1 * e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        return p_noisy, e_rand

    def denoise(self, p_t, eps_p, mask_generate, t):
        alpha = self.var_sched.alphas[t].clamp_min(self.var_sched.alphas[-2])
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1, 1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1, 1)

        mu = c0 * (p_t - c1 * eps_p)
        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )
        p_next = mu + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)

        return mu, p_next, z


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer('_dummy', torch.empty([0]))

    def add_noise(self, v_0, mask_generate, t):
        """
        Args:
            v_0: Orientations, (N, L, 3).
            mask_generate: Generated residue mask, (N, L).
            t: Time steps, (N,).
        Returns:
            v_noisy: Noisy orientations at step t, (N, L, 3).
            e_scaled: Scaled SO(3) noise, (N, L, 3).
        """
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_scaled = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device
        )
        E_scaled = so3vec_to_rotation(e_scaled)
        R0_scaled = so3vec_to_rotation(c0 * v_0)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        N, L = mask_generate.size()
        e = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device
        )
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e),
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next


class AminoacidCategoricalTransition(nn.Module):

    def __init__(self, num_steps, num_classes=20, var_sched_opt={}, prior_b=0):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
        self.prior_b = prior_b
        self.register_buffer(
            'hydropathy_prob',
            torch.tensor(hydropathy_prob, dtype=torch.float),
            persistent=False,
        )

    @staticmethod
    def _sample(c):
        """
        Args:
            c: Categorical probabilities, (N, L, K).
        Returns:
            x: Sampled class indices, (N, L).
        """
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def _hydropathy_prior(self, device, dtype, prior_b):
        prior_b = float(prior_b)
        if prior_b < 0.0 or prior_b > 1.0:
            raise ValueError(f"prior_b must be in [0, 1], got {prior_b}.")
        prior = self.hydropathy_prob.to(device=device, dtype=dtype)
        return prior_b * prior + (1 - prior_b) / self.num_classes

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0: Initial categories, (N, L).
            mask_generate: Generated residue mask, (N, L).
            t: Time steps, (N,).
        Returns:
            c_t: Noisy categorical probabilities, (N, L, K).
            x_t: Sampled noisy categories, (N, L).
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]
        c_noisy = alpha_bar * c_0 + (1 - alpha_bar) / K
        c_t = torch.where(mask_generate[..., None].expand(N, L, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t, prior_b=None):
        """
        Args:
            x_t: Current categories or probabilities, (N, L) or (N, L, K).
            x_0: Denoised categories or probabilities, (N, L) or (N, L, K).
            t: Time steps, (N,).
            prior_b: Optional hydropathy-prior weight. If None, use a uniform prior.
        Returns:
            theta: Posterior probabilities at step t-1, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()

        if x_0.dim() == 3:
            c_0 = x_0
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()

        alpha = self.var_sched.alpha_bars[t][:, None, None]
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]

        if prior_b is None:
            theta = ((alpha * c_t) + (1 - alpha) / K) * ((alpha_bar * c_0) + (1 - alpha_bar) / K)
        else:
            prior = self._hydropathy_prior(t.device, c_t.dtype, prior_b)
            theta = ((alpha * c_t) + (1 - alpha) * prior) * ((alpha_bar * c_0) + (1 - alpha_bar) * prior)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t, prior_b=None):
        """
        Args:
            x_t: Current categories, (N, L).
            c_0_pred: Predicted denoised probabilities, (N, L, K).
            mask_generate: Generated residue mask, (N, L).
            t: Time steps, (N,).
            prior_b: Optional hydropathy-prior weight. If None, use a uniform prior.
        Returns:
            post: Posterior probabilities at step t-1, (N, L, K).
            x_next: Sampled categories at step t-1, (N, L).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()
        post = self.posterior(c_t, c_0_pred, t=t, prior_b=prior_b)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next
