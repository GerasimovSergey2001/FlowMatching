import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class FlowMatching(nn.Module):
    def __init__(self, model, obs_dim=(2,), sigma_min=1e-6, n_samples=10):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.obs_dim = obs_dim
        self.n_samples = n_samples
    
    def process_timesteps(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        if t.shape[0]!=x.shape[0] or len(t.shape)!=1:
            raise ValueError("Timesteps shape should (batch_size, )")
        return t

    def forward(self, t, x0, x1):
        """ 
        Computes velocity v from the equation dphi(t, x) = v(t, phi(t, x))dt. 
        """
        t = self.process_timesteps(t, x0)
        x = self.conditional_flow(t, x0, x1)
        return self.model(t, x)

    def velocity(self, t, x0):
        t = self.process_timesteps(t, x0)
        return self.model(t, x0)
    
    def reversed_velocity_with_div(self, t, state):
        s = 1-t
        x, logp = state
        x_ = x.detach().clone().requires_grad_(True)
        div_estimates = []
        with torch.set_grad_enabled(True):
            for i in range(self.n_samples):
                v = self.model(s, x_)
                is_last = (i == self.n_samples - 1)
                div_estimates.append(
                    self.approx_div(v, x_, retain_graph=not is_last)
                )
        
        mean_div = torch.stack(div_estimates).mean(dim=0)
        return ((-v).detach(), mean_div)

    def conditional_flow(self, t, x0, x1):
        """
        Computes \phi(t,x) = \sigma(t, x1)x + \mu(t, x1), where \phi(0,x) = x = x0
        \sigma(t, x1) = 1 - (1-\sigma_{min})t
        \mu(t, x1) = t*x1
        
        :param t: timestep. Float in [0,1].
        :param x0: starting point sampled from N(0, I).
        :param x1: observation
        """
        dims = [1]*(len(x0.shape)-1)
        t = t.view(-1, *dims)
        return (1 - (1-self.sigma_min)*t)*x0 + t*x1
    
    def criterion(self, t, x0, x1):
       v = self.forward(t, x0, x1)
       target = x1 - (1-self.sigma_min)*x0
       dim = tuple(torch.arange(1, len(x0.shape)))
       return torch.mean((v - target).pow(2).sum(dim=dim))
    
    def sample(self, n_samples, method='dopri5', rtol=1e-5, atol=1e-5):
        x0 = torch.randn([n_samples]+list(self.obs_dim))
        t = torch.linspace(0,1,2)
        with torch.no_grad():
            return odeint(self.model, x0, t, rtol=rtol, atol=atol, method=method)[-1,:,:]
        

    def approx_div(self, f_x, x, retain_graph=True):
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=True, retain_graph=retain_graph)[0]
        return (e_dzdx*z).view(z.shape[0], -1).sum(dim=1)

        
    def logp(self, x1, n_samples=50, rtol=1e-05, atol=1e-05):
        self.n_samples = n_samples
        t = torch.linspace(0, 1, 2)
        phi, f = odeint(
            self.reversed_velocity_with_div, 
            (x1, torch.zeros((x1.shape[0], 1))), 
            t, 
            rtol=rtol,
            atol=atol,
            adjoint_params = self.model.parameters(),
            )
        phi, f = phi[-1].detach().cpu(), f[-1].detach().cpu().flatten()
        logp_noise = -0.5 * (phi.pow(2).sum(1) + phi.shape[1] * torch.log(torch.tensor(2 * torch.pi)))
        return logp_noise - f
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.model.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
        