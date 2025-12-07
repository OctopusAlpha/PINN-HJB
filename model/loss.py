import torch
import torch.autograd as autograd

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
    r = x[:, 1]  # 假设 r 是 x 的第
    
    '''Aa的计算，即随机利率理论'''
    # Bt = (1-torch.exp(-k*t))/k
    # At = torch.exp((Bt-(T-t))*(k*k*sigemar-sigemar*sigemar/2)/(k*k) - (sigemar**2*Bt*Bt/4/k))
    Aa = 1
    theta = 0.7
    rou = 0.5
    # ran = torch.randn(40,device=device) * torch.sqrt(t)
    t1 = t.view(-1, 1)  # 或者 .unsqueeze(1)

    '''计算phi通过构建随机变量S和mu,sigma，即随机贴现率'''
    ran = torch.randn(x.shape[0], 40, device=device) * torch.sqrt(t1)
    t2 = t.unsqueeze(1) 
    # ran = torch.randn(128, 40, device=device) * torch.sqrt(t).expand(-1, 40)
    exp_term = torch.exp((mu - 0.5 * sigma**2) * t2 + sigma * ran)  # 自动广播成 (128,40)
    S1 = S * 0.5 * (1 + exp_term)
    S2 = S1 ** 2
    phi = torch.exp ( -(1-theta)**-1 *(0.2 + 0.2*S1 +0.2*S2 ))
    # print('phi shape:', phi.shape)
    phi = phi.mean(dim=1, keepdim=True)  # (128, 1)
    # print('phi shape:', phi.shape)
    '''计算最优解Cc'''
    
    # print(V_w)
    Cc = (V_w)**(theta-1)*phi





    # print(r.shape)
    # print(w.shape)
    # print(V_t.shape)
    # print(Cc.shape)
    '''计算 HJB 残差''' 
    sigemar = 0.04
    # print(f"mu shape: {mu.view(-1, 1).expand_as(V_t).shape}")
    M = V_t 
    + V_w *w*( pi@mu + ( 1 - pi.sum(dim=1)) * r - Cc) 
    + 0.5* sigemar**2 *V_rr 
    +  V_ww * (
        0.5*(pi * sigma).sum()**2 
        + 0.5*((1 - pi.sum(dim=1))* w *Aa)**2 
        +(1-pi.sum(dim=1) )*w**2 
        )
    +  0.5 * V_wr * rou * sigma.sum()* ( (1-pi.sum(dim=1) )*w )

    M = abs(M*10)

    '''计算 边界损失''' 
    phi1 = torch.clamp(phi, min=1e-3)

    tt = theta **-1 * phi1 **(1-theta) * w**theta * (-6.676 - 0.8* 1.198 *3**2 +5.705* 3)
    # print((-6.676*t + 0.8* -1.198 *r**2 +7.705* r)**2 )
    # print(theta **-1 * phi1 **(1-theta) * w**theta * (-6.676 - 0.8* 1.198 *r**2 +5.705* r) )
    # phi1 = torch.clamp(phi, min=1e-3)
    tt =  tt.mean(dim=1, keepdim=True)  # (128, 1)
    # print('phi1 shape:', tt.shape)
    # print('V shape:', V.shape)
    boundary_loss = (V - tt).mean()  
    boundary_loss = abs(boundary_loss)

    '''计算 pi 约束 loss''' #还要固定其最大以及最小值
    pi_sigma = torch.sum(pi * sigma, dim=1, keepdim=True)  # [batch_size, 1]
    rhs = 0.5 * V_wr.unsqueeze(1) * x_scalar  # [batch_size, 1]
    constraint_loss = ((pi_sigma - rhs) ** 2).mean()
    constraint_loss = abs(100*constraint_loss) 
    if constraint_loss.item() == "nan":
        print(constraint_loss)

    # 可加更多损失项，比如 HJB残差、边界条件等
    return constraint_loss + M.mean() , constraint_loss, M.mean(), V, pi


def bc_loss(model,x,S,sigma,mu,device):
    t = x[:, 0:1]
    w = x[:, 2:3]  # 假设 w 是 x 的第 3 个变量
    r = x[:, 1:2]  # 假设 r 是 x 的第 2 个变量

    batch_size = x.shape[0]
    num_mc = 40
    t_exp = t.expand(-1, num_mc)  # (128,40)

    # t2 = t.unsqueeze(1) 
    # t1 = t.view(-1, 1)
    # ran = torch.randn(x.shape[0], 40, device=device) * torch.sqrt(t1)
    ran = torch.randn(x.shape[0], 40, device=device) 
    output = model(x)
    V = output[:, 0:1] 
    theta = 0.7
    exp_term = torch.exp((mu - 0.5 * sigma**2) * t_exp + sigma * ran)  # 自动广播成 (128,40)
    S1 = S * 0.5 * (1 + exp_term)
    S2 = S1 ** 2

    phi = torch.exp ( (0.2 + 0.2*S1 +0.2*S2 )/(theta-1))
    # print('phi shape:', phi.shape)
    phi = phi.mean(dim=1, keepdim=True)

    '''计算 边界损失''' 
    phi1 = torch.clamp(phi, min=1e-3,max=1e6)

    tt = theta **-1 *phi1 **(1-theta) * w**theta * (-6.676 - 0.8* 1.198 *r**2 +5.705* r)
    # print(phi1.shape)
    # print(tt.shape)
    # print(r.shape)
    # print(V.shape)
    # tt =  tt.mean(dim=1, keepdim=True)  # (128, 1)
    # print('phi1 shape:', tt.shape)
    # print('V shape:', V.shape)
    # boundary_loss = (V - tt).mean()  
    # boundary_loss = abs(boundary_loss)
    boundary_loss = torch.mean((V - tt)**2)
    
    return boundary_loss