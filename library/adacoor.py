import torch

class AdaCoor(torch.optim.Optimizer):
    def __init__(self, params, eps=1e-8, vt_rate=1.00, *args, **kwargs):
        defaults = dict(epsilon=eps, lr=1.0, vt_rate=vt_rate)
        super(AdaCoor, self).__init__(params, defaults)

    def scale_gradients(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.grad.mul_(group['lr'])  # scale gradients by learning rate

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.scale_gradients()  # scale gradients before optimization step

        vt_total = 0.0  # total of all vt values
        num_params = 0  # number of parameters

        for group in self.param_groups:
            with torch.no_grad():
                # Initialize epsilon and vt_rate as tensors
                epsilon = torch.tensor([group['epsilon']], dtype=torch.bfloat16, device=next(iter(group['params'])).device)
                vt_rate = torch.tensor([group['vt_rate']], dtype=torch.bfloat16, device=next(iter(group['params'])).device)

                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Initialize state variable for vt
                    if 'vt' not in state:
                        state['vt'] = torch.zeros_like(p.data, device=p.device).to(dtype=torch.bfloat16, device=p.device)

                    vt = state['vt']
                    vt.add_((vt_rate * epsilon * p.grad.data ** 2).to(dtype=torch.bfloat16, device=p.device))

                    # Add to the total vt and increment the parameter count
                    vt_total += vt.mean().item()
                    num_params += 1

                    gt_hat = (epsilon * p.grad.data).to(dtype=torch.float32, device=p.device)

                    denom = vt.sqrt().add_(group['epsilon']).to(dtype=p.dtype, device=p.device)
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)

        # Compute and print the global average of vt
        vt_global_avg = vt_total / num_params
        #print("Global average of vt: ", vt_global_avg)

        return loss