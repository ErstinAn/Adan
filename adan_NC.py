import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class adan_NC(Optimizer):
    """
    Implements Topology-Aware AdamW with NC Boost.
    Based on the provided pseudocode and modified from Adan's structure.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for
            first-order momentum, second-order momentum, and nc_boost. (default: (0.9, 0.999, 0.9))
        eps (float, optional): term added to the denominator to improve numerical stability. (default: 1e-8)
        alpha (float, optional): base trajectory coefficient. (default: 1.0)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 0.0 no clip)
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999, 0.9),  # 对应伪代码的 beta1, beta2, beta_a
                 eps=1e-8,
                 alpha=1.0,                # 新增：base trajectory coefficient
                 weight_decay=0.0,
                 max_grad_norm=0.0):
        if not 0.0 <= max_grad_norm:
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameters')

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        alpha=alpha,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # 按照伪代码第一步初始化所有状态为 0
                    state['exp_avg'] = torch.zeros_like(p)      # m_0 = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)   # v_0 = 0
                    state['nc_boost'] = torch.zeros_like(p)     # a_0 = 0
                    state['prev_grad'] = torch.zeros_like(p)    # g_0 = 0
                    state['prev_param'] = p.clone().detach()    # theta_{-1} = theta_0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 保留原有的全局梯度裁剪逻辑
        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)
            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())
            global_grad_norm = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group['eps']),
                max=1.0).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            nc_boosts = []
            prev_grads = []
            prev_params = []

            beta1, beta2, beta_a = group['betas']
            
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # Adam 的偏差校正系数 (因为伪代码里使用的是除以 1-beta^k)
            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # 按照伪代码第一步的初始化逻辑
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)      # m_0 = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)   # v_0 = 0
                    state['nc_boost'] = torch.zeros_like(p)     # a_0 = 0
                    state['prev_grad'] = torch.zeros_like(p)    # g_0 = 0
                    state['prev_param'] = p.clone().detach()    # theta_{-1} = theta_0

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                nc_boosts.append(state['nc_boost'])
                prev_grads.append(state['prev_grad'])
                prev_params.append(state['prev_param'])

            if not params_with_grad:
                continue

            # 调用核心计算函数
            _single_tensor_ta_adamw(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                nc_boosts=nc_boosts,
                prev_grads=prev_grads,
                prev_params=prev_params,
                beta1=beta1,
                beta2=beta2,
                beta_a=beta_a,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                alpha=group['alpha'],
                clip_global_grad_norm=clip_global_grad_norm,
            )

        return loss


def _single_tensor_ta_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    nc_boosts: List[Tensor],
    prev_grads: List[Tensor],
    prev_params: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta_a: float,
    bias_correction1: float,
    bias_correction2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    alpha: float,
    clip_global_grad_norm: float,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        nc_boost = nc_boosts[i]
        prev_grad = prev_grads[i]
        prev_param = prev_params[i]

        # 裁剪当前梯度
        grad.mul_(clip_global_grad_norm)

        # Step 4: a_k = beta_a * a_{k-1} + (1 - beta_a)(g_k - g_{k-1})
        delta_g = grad - prev_grad
        nc_boost.mul_(beta_a).add_(delta_g, alpha=1 - beta_a)

        # Step 5: \Delta \theta_k = \theta_k - \theta_{k-1}
        delta_theta = param - prev_param
        delta_theta_sq_norm = delta_theta.pow(2).sum()

        # Step 6: 计算 \kappa_k
        if delta_theta_sq_norm == 0:
            # 解决第一步(k=1)时 theta_0 - theta_{-1} = 0 导致的除零错误
            kappa = 0.0 
        else:
            # \kappa_k = <\Delta \theta_k, (g_k - g_{k-1})> / ||\Delta \theta_k||^2
            kappa = torch.dot(delta_theta.flatten(), delta_g.flatten()) / delta_theta_sq_norm
            kappa = kappa.item() # 转换为 Python float

        # Step 7: \alpha_k = \alpha * (1 + tanh(-\kappa_k))
        alpha_k = alpha * (1 + math.tanh(-kappa))

        # 辅助变量：增强型有效梯度 g_k + \alpha_k * a_k
        eff_grad = grad + nc_boost * alpha_k

        # Step 8: m_k = \beta_1 m_{k-1} + (1 - \beta_1)(g_k + \alpha_k a_k)
        exp_avg.mul_(beta1).add_(eff_grad, alpha=1 - beta1)

        # Step 9: v_k = \beta_2 v_{k-1} + (1 - \beta_2)[g_k + \alpha_k a_k]^2
        exp_avg_sq.mul_(beta2).addcmul_(eff_grad, eff_grad, value=1 - beta2)

        # Step 10 & 11: 偏差校正
        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2

        # Step 12: 参数更新 AdamW style: \theta = \theta - lr * (m_hat / (sqrt(v_hat) + eps) + \lambda \theta)
        if weight_decay > 0:
            # AdamW 的 Weight Decay 是直接作用在参数上的
            param.mul_(1 - lr * weight_decay)
        
        denom = v_hat.sqrt().add_(eps)
        param.addcdiv_(m_hat, denom, value=-lr)

        # 结尾：更新历史状态以供下一步使用
        prev_grad.copy_(grad)
        prev_param.copy_(param)