import torch
import torch.autograd as autograd

# 常量定义
THETA = 0.7
ROU = 0.5
SIGEMAR = 0.04
AA = 1
NUM_MC = 40

def _compute_phi(S, mu, sigma, t, device, use_sqrt_t=True):
    """
    计算phi（随机贴现率）
    
    Args:
        S: 股票价格张量 [batch_size, n_assets]
        mu: 漂移率 [n_assets]
        sigma: 波动率 [n_assets]
        t: 时间 [batch_size] 或 [batch_size, 1]
        device: 设备
        use_sqrt_t: 是否在随机数中使用sqrt(t)，pinn_loss使用True，bc_loss使用False
    
    Returns:
        phi: [batch_size, 1]
    """
    batch_size = S.shape[0]
    if t.dim() == 1:
        t_exp = t.unsqueeze(1).expand(-1, NUM_MC)  # [batch_size, 40]
        t_unsqueezed = t.unsqueeze(1)
    else:
        t_exp = t.expand(-1, NUM_MC)
        t_unsqueezed = t
    
    if use_sqrt_t:
        ran = torch.randn(batch_size, NUM_MC, device=device) * torch.sqrt(t_unsqueezed)
    else:
        ran = torch.randn(batch_size, NUM_MC, device=device)
    
    exp_term = torch.exp((mu - 0.5 * sigma**2) * t_exp + sigma * ran)
    S1 = S * 0.5 * (1 + exp_term)
    S2 = S1 ** 2
    # 公式等价: -(1-theta)**-1 = 1/(theta-1)
    phi = torch.exp(-(1-THETA)**-1 * (0.2 + 0.2*S1 + 0.2*S2))
    phi = phi.mean(dim=1, keepdim=True)
    return phi

def _compute_boundary_value(phi, w, r, theta=THETA):
    """
    计算边界值tt
    
    Args:
        phi: [batch_size, 1]
        w: [batch_size] 或 [batch_size, 1]
        r: [batch_size] 或 [batch_size, 1]
        theta: 风险厌恶参数
    
    Returns:
        tt: [batch_size, 1]
    """
    if w.dim() == 1:
        w = w.unsqueeze(1)
    if r.dim() == 1:
        r = r.unsqueeze(1)
    
    phi1 = torch.clamp(phi, min=1e-3, max=1e6)
    # pinn_loss中使用固定值3，bc_loss中使用r
    # 这里使用r，因为bc_loss中需要r
    tt = theta**-1 * phi1**(1-theta) * w**theta * (-6.676 - 0.8*1.198*r**2 + 5.705*r)
    return tt

def pinn_loss(model, x, sigma,mu, x_scalar,S,device):
    # x: [batch_size, input_dim]
    x.requires_grad_(True)
    output = model(x)  # [batch_size, 1 + 40]
    V = output[:, 0:1] 
    # V = torch.abs(V) # [batch_size, 1]
    ''' 输出的pi只能为非负且和为1'''
    raw_pi = output[:, 1:]  # [batch_size, 40]
    pi = torch.softmax(raw_pi, dim=1) 
    pi = pi [:,:-1]   
    # print(pi.shape)          # 保证权重合理
    # print(output)
    # --------- 计算 V_wr ---------
    # 对 w 求导 (假设 w 是 x 的第 i 个变量，比如 x[:, 3])
    grads = autograd.grad(V, x, grad_outputs=torch.ones_like(V),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]  # [batch, input_dim]
    V_t = grads[:, 0]  # 对 t 偏导
    V_w = grads[:, 2]  # 对 w 偏导
    V_r = grads[:, 1]  # 对 r 偏导
    V_w = torch.nn.functional.softplus(V_w) + 1e-6
    # 对 w 和 r 的二阶导
    V_wr = autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                         create_graph=True, retain_graph=True, only_inputs=True)[0][:, 1]  # d²V/dwd r
    V_ww = autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                         create_graph=True, retain_graph=True, only_inputs=True)[0][:, 2]  # d²V/dwd w
    V_rr = autograd.grad(V_r, x, grad_outputs=torch.ones_like(V_r),
                         create_graph=True, retain_graph=True, only_inputs=True)[0][:, 1]  # d²V/drd r
    t = x[:, 0]  # 假设 t 是 x 的第 0 个变量
    w = x[:, 2]  # 假设 w 是 x 的第 3 个变量
    r = x[:, 1]  # 假设 r 是 x 的第 1 个变量
    
    '''计算phi通过构建随机变量S和mu,sigma，即随机贴现率'''
    phi = _compute_phi(S, mu, sigma, t, device, use_sqrt_t=True)
    
    '''计算最优解Cc'''
    Cc = (V_w)**(THETA-1)*phi





    '''计算 HJB 残差''' 
    M = (V_t 
         + V_w * w * (pi @ mu + (1 - pi.sum(dim=1)) * r - Cc) 
         + 0.5 * SIGEMAR**2 * V_rr 
         + V_ww * (
             0.5 * (pi * sigma).sum()**2 
             + 0.5 * ((1 - pi.sum(dim=1)) * w * AA)**2 
             + (1 - pi.sum(dim=1)) * w**2 
         )
         + 0.5 * V_wr * ROU * sigma.sum() * ((1 - pi.sum(dim=1)) * w))

    M = abs(M * 10)

    '''计算 边界损失（pinn_loss中使用固定r=3）''' 
    r_fixed = torch.full_like(r, 3.0)
    tt = _compute_boundary_value(phi, w, r_fixed, THETA)
    boundary_loss = abs((V - tt).mean())

    '''计算 pi 约束 loss''' #还要固定其最大以及最小值
    pi_sigma = torch.sum(pi * sigma, dim=1, keepdim=True)  # [batch_size, 1]
    rhs = 0.5 * V_wr.unsqueeze(1) * x_scalar  # [batch_size, 1]
    constraint_loss = ((pi_sigma - rhs) ** 2).mean()
    constraint_loss = abs(100*constraint_loss) 
    if constraint_loss.item() == "nan":
        print(constraint_loss)

    # 可加更多损失项，比如 HJB残差、边界条件等
    return constraint_loss + M.mean() , constraint_loss, M.mean(), V, pi


def bc_loss(model, x, S, sigma, mu, device):
    """
    计算边界损失
    
    Args:
        model: PINN模型
        x: 输入 [batch_size, input_dim]
        S: 股票价格 [batch_size, n_assets]
        sigma: 波动率 [n_assets]
        mu: 漂移率 [n_assets]
        device: 设备
    
    Returns:
        boundary_loss: 边界损失标量
    """
    t = x[:, 0]  # 时间
    w = x[:, 2]  # 财富
    r = x[:, 1]  # 利率
    
    output = model(x)
    V = output[:, 0:1]
    
    # 计算phi（bc_loss中不使用sqrt(t)）
    phi = _compute_phi(S, mu, sigma, t, device, use_sqrt_t=False)
    
    # 计算边界损失（使用r而非固定值）
    tt = _compute_boundary_value(phi, w, r, THETA)
    boundary_loss = torch.mean((V - tt)**2)
    
    return boundary_loss