import torch

def _generate_base_data(batch_size, device, t_value=None):
    """
    生成基础数据（t, r, w）
    
    Args:
        batch_size: 批次大小
        device: 设备
        t_value: 时间值，如果为None则随机生成U(0,1)，否则使用固定值
    
    Returns:
        torch.Tensor: [batch_size, 3] (t, r, w)
    """
    if t_value is None:
        t = torch.rand((batch_size, 1), device=device)  # t ~ U(0,1)
    else:
        t = torch.full((batch_size, 1), t_value, device=device)
    
    r = (torch.randn((batch_size, 1), device=device) + 3) / 100  # r ~ N(3,0.01)
    w = torch.rand((batch_size, 1), device=device) + 0.5  # w ~ U(0.5,1.5)
    
    return torch.concat([t, r, w], dim=1)

def generate_domain_data(batch_size, device):
    """生成域内数据"""
    return _generate_base_data(batch_size, device, t_value=None)

def generate_boundary_data(batch_size, device):
    """生成边界数据（t=1）"""
    return _generate_base_data(batch_size, device, t_value=1.0)