import torch
def generate_domain_data(batch_size,device) :
    return  torch.concat(
                [ 
                torch.rand((batch_size, 1), device=device),   # t ~ U(0,1)
                (torch.randn((batch_size, 1), device= device) + 3)/100, # r ~ N(3,0.01)
                torch.rand((batch_size, 1), device= device) + 0.5, # w ~ U(0.5,1.5)
                ],
                dim=1
            )
    
def generate_boundary_data(batch_size,device) :
    
    return torch.concat(
            [ 
            torch.ones((batch_size, 1), device=device),   # t ~ U(0,1)
            (torch.randn((batch_size, 1), device= device) + 3)/100, # r ~ N(3,0.01)
            torch.rand((batch_size, 1), device= device) + 0.5, # w ~ U(0.5,1.5)
            ],
            dim=1
        )