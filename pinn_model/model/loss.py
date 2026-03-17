import torch
import torch.autograd as autograd
import math

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
    
    # S: [batch_size, n_assets] -> 先对资产维度求均值，得到 [batch_size, 1]，再广播到 [batch_size, NUM_MC]
    S_mean = S.mean(dim=1, keepdim=True)  # [batch_size, 1]
    exp_term = torch.exp((mu - 0.5 * sigma**2).mean() * t_exp + sigma.mean() * ran)  # [batch_size, NUM_MC]
    S1 = S_mean * 0.5 * (1 + exp_term)   # [batch_size, NUM_MC]
    S2 = S1 ** 2
    # 公式等价: -(1-theta)**-1 = 1/(theta-1)
    phi = torch.exp(-(1-THETA)**-1 * (0.2 + 0.2*S1 + 0.2*S2))
    phi = phi.mean(dim=1, keepdim=True)   # [batch_size, 1]
    return phi

def _compute_boundary_value(phi, t,w, r, theta=THETA):
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
    if t.dim() == 1:
        t = t.unsqueeze(1)


    phi1 = torch.clamp(phi, min=1e-3, max=1e6)
    # pinn_loss中使用固定值3，bc_loss中使用r
    # 这里使用r，因为bc_loss中需要r
    # tt = theta**-1 * phi1**(1-theta) * w**theta * torch.exp(-6.676 - 0.8*1.198*r**2 + 5.705*r)
    # tt = theta**-1 * phi1**(1-theta) * w**theta * torch.exp(-9.01  -1.24*r*r + 0.8*r)
    tt = theta**-1 * phi1**(1-theta) * w**theta * torch.exp(-27.01  -10.24*r*r +5*r)
    return tt

def pinn_loss(model, x, sigma,mu, x_scalar,S,device):
    # x: [batch_size, input_dim]
    x.requires_grad_(True)
    output = model(x)  # [batch_size, output_dim] = [batch_size, 42]
    h = output[:, 0:1] 
    x_safe = torch.clamp(x, min=1e-3) 
    
    # 【安全锁 2】：限制 h 的范围，防止 torch.exp(h) 数值爆炸引发 Inf
    # exp(10) 大约是 22000，exp(-20) 极小，对浮点数是绝对安全的范围
    h_safe = torch.clamp(h, min=-20.0, max=10.0)
    V = (1.0 / THETA) * (x_safe ** THETA) * torch.exp(h_safe)
    # V = torch.abs(V) # [batch_size, 1]
    ''' 输出的pi只能为非负且和为1'''
    raw_pi = output[:, 1:]  # [batch_size, 41]
    # pi = torch.softmax(raw_pi, dim=1) 
    pi = raw_pi
    pi = pi [:,:-1]   # [batch_size, 40] 去掉最后一个，保留40个资产的权重   
    # print(pi.shape)          # 保证权重合理
    # print(output)
    # --------- 计算 V_wr ---------
    # 对 w 求导 (假设 w 是 x 的第 i 个变量，比如 x[:, 3])
    grads = autograd.grad(V, x, grad_outputs=torch.ones_like(V),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]  # [batch, input_dim]
    V_t = grads[:, 0]  # 对 t 偏导
    V_w = grads[:, 2]  # 对 w 偏导
    V_r = grads[:, 1]  # 对 r 偏导
    # V_w 不做 softplus 处理，保留真实梯度用于 HJB 二阶导计算
    # 仅在数值计算时加一个小量防除零，不改变计算图
    V_w = V_w + 1e-6
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
    # pi 是 [batch_size, n_pi_assets]，需要切片 mu 和 sigma 以匹配 pi 的维度
    n_pi_assets = pi.shape[1]  # pi 的实际维度
    mu_pi = mu[:n_pi_assets]  # 匹配 pi 的维度
    sigma_pi = sigma[:n_pi_assets]  # 匹配 pi 的维度
    
    M = (V_t 
         + V_w * w * (pi @ mu_pi + (1 - pi.sum(dim=1)) * r - Cc) 
         + 0.5 * SIGEMAR**2 * V_rr 
         + V_ww * (
             0.5 * (pi * sigma_pi).sum(dim=1)**2 
             + 0.5 * ((1 - pi.sum(dim=1)) * w * AA)**2 
             + (1 - pi.sum(dim=1)) * w**2 
         )
         + 0.5 * V_wr * ROU * sigma.sum() * ((1 - pi.sum(dim=1)) * w))

    # M = abs(M * 10)
    M = abs(M)
    '''计算 pi 约束 loss''' #还要固定其最大以及最小值
    # sigma_pi 已在上面定义，直接使用（维度已匹配）
    pi_sigma = torch.sum(pi * sigma_pi, dim=1, keepdim=True)  # [batch_size, 1]
    rhs = 0.5 * V_wr.unsqueeze(1) * x_scalar  # [batch_size, 1]
    constraint_loss = ((pi_sigma - rhs) ** 2).mean()
    constraint_loss = abs(constraint_loss)  # 去掉×100，避免主导训练
    if math.isnan(constraint_loss.item()):
        print(f"constraint_loss is NaN: {constraint_loss}")

    # 可加更多损失项，比如 HJB残差、边界条件等
    # 返回各项分离，由 train.py 统一加权
    return constraint_loss, M.mean(), V, pi


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
    # V = output[:, 0:1]
    h = output[:, 0:1]
    x_safe = torch.clamp(x, min=1e-3) 
    
    # 【安全锁 2】：限制 h 的范围，防止 torch.exp(h) 数值爆炸引发 Inf
    # exp(10) 大约是 22000，exp(-20) 极小，对浮点数是绝对安全的范围
    h_safe = torch.clamp(h, min=-20.0, max=10.0)
    V = (1.0 / THETA) * (x_safe ** THETA) * torch.exp(h_safe)
    
    # 计算phi（bc_loss中不使用sqrt(t)）
    phi = _compute_phi(S, mu, sigma, t, device, use_sqrt_t=False)
    
    # 计算边界损失（使用r而非固定值）
    tt = _compute_boundary_value(phi, t, w, r, THETA)
    # logger.info(f"  Boundary Loss: { tt.mean().item():.6f}")
    boundary_loss = torch.mean((V - tt)**2)
    
    return boundary_loss,tt.mean().item()

def smooth_loss(model, x, S, sigma, mu, device):
    """
    平滑性损失函数：计算投资组合权重 π 对 w 和 r 的偏导数的平方和
    
    公式:
    $$\mathcal{L}_{smooth} = \frac{1}{N_{data}} \sum_{j=1}^{N_{data}} \sum_{i=1}^n \left( \left| \frac{\partial \pi_i}{\partial w} \right|^2 + \left| \frac{\partial \pi_i}{\partial r} \right|^2 \right)$$
    
    Args:
        model: PINN模型
        x: 输入 [batch_size, input_dim]
        S: 股票价格 [batch_size, n_assets]
        sigma: 波动率 [n_assets]
        mu: 漂移率 [n_assets]
        device: 设备
    
    Returns:
        smooth_loss: 平滑性损失标量
    """
    x.requires_grad_(True)
    output = model(x)  # [batch_size, output_dim]
    
    # 提取投资组合权重 π (通过softmax归一化)
    raw_pi = output[:, 1:]  # [batch_size, 41]
    # pi = torch.softmax(raw_pi, dim=1)
    pi = raw_pi
    pi = pi[:, :-1]  # [batch_size, 40] 去掉最后一个，保留40个资产的权重
    
    # 计算 π 对 w 的偏导数 (w 是 x 的第 2 个变量)
    grad_pi_w = []
    for i in range(pi.shape[1]):
        grad_w = autograd.grad(pi[:, i], x, grad_outputs=torch.ones_like(pi[:, i]),
                               create_graph=True, retain_graph=True, only_inputs=True)[0][:, 2]
        grad_pi_w.append(grad_w)
    grad_pi_w = torch.stack(grad_pi_w, dim=1)  # [batch_size, 40]
    
    # 计算 π 对 r 的偏导数 (r 是 x 的第 1 个变量)
    grad_pi_r = []
    for i in range(pi.shape[1]):
        grad_r = autograd.grad(pi[:, i], x, grad_outputs=torch.ones_like(pi[:, i]),
                               create_graph=True, retain_graph=True, only_inputs=True)[0][:, 1]
        grad_pi_r.append(grad_r)
    grad_pi_r = torch.stack(grad_pi_r, dim=1)  # [batch_size, 40]
    
    # 计算平滑性损失: 偏导数的平方和
    loss_smooth = (grad_pi_w ** 2 + grad_pi_r ** 2).mean() * 0.1  # 原为×100000，降低平滑损失权重
    
    return loss_smooth

def consistency_loss(model, x, S, sigma, mu, device):
    """
    一致性损失函数：处理42维输出
    
    输出结构:
    - 第1个: 值函数 V
    - 第2-41个: 40个资产比例，必须为正数，加总在[0.6, 0.95]之间
    - 第42个: 现金比例，必须为正数
    - 第2-42个(共41个): 加总等于1
    
    Args:
        model: PINN模型
        x: 输入 [batch_size, input_dim]
        S: 股票价格 [batch_size, n_assets]
        sigma: 波动率 [n_assets]
        mu: 漂移率 [n_assets]
        device: 设备
    
    Returns:
        total_loss: 总损失
        loss_details: 各损失项的字典
    """
    x.requires_grad_(True)
    output = model(x)  # [batch_size, 42]
    
    # 提取原始输出
    V = output[:, 0:1]  # 值函数 [batch_size, 1]
    raw_pi = output[:, 1:42]  # 原始41个比例输出 [batch_size, 41]
    
    # 分离40个资产比例和1个现金比例
    pi_assets = raw_pi[:, :40]  # [batch_size, 40] 资产比例
    pi_cash = raw_pi[:, 40:41]  # [batch_size, 1] 现金比例
    
    # 计算损失项
    losses = {}
    
    # 1. 正数约束损失: 所有41个比例必须为正数
    # 对负数部分施加惩罚
    losses['positivity'] = (torch.clamp(-raw_pi, min=0.0) ** 2).mean()
    
    # 2. 资产比例范围约束损失: 40个资产比例之和应在[0.6, 0.95]
    assets_sum = pi_assets.sum(dim=1)  # [batch_size]
    # 下界约束: 如果总和小于0.6，产生损失
    lower_violation = torch.clamp(0.6 - assets_sum, min=0.0)
    # 上界约束: 如果总和大于0.95，产生损失
    upper_violation = torch.clamp(assets_sum - 0.95, min=0.0)
    losses['assets_range'] = (lower_violation ** 2 + upper_violation ** 2).mean()
    
    # 3. 总和约束: 第2-42个(41个)加总等于1
    total_41_sum = raw_pi.sum(dim=1)  # [batch_size]
    losses['sum_to_one'] = ((total_41_sum - 1.0) ** 2).mean()
    
    # 总损失 (增加正数约束权重，确保模型学习输出正数)
    total_loss = (10.0 * losses['positivity'] + 
                  1.0 * losses['assets_range'] + 
                  1.0 * losses['sum_to_one'])
    
    # loss_details = {
    #     'total': total_loss.item(),
    #     'normalization': losses['normalization'].item(),
    #     'assets_range': losses['assets_range'].item(),
    #     'positivity': losses['positivity'].item(),
    #     'sum_to_one': losses['sum_to_one'].item(),
    #     'pi_assets_sum_mean': assets_sum.mean().item(),
    #     'pi_cash_mean': pi_cash.mean().item()
    # }
    
    # return total_loss, loss_details, V, pi_assets, pi_cash
    return total_loss

    
