import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
from contextlib import contextmanager
import abc

# Set the Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

def get_score_fn(sde, diffusion_model):
    def score_fn(x, y, t):
        noise_prediction = diffusion_model(x, y, t)
        _, std = sde.marginal_prob(x, t)
        std = std.view(std.shape[0], *[1 for _ in range(len(x.shape) - 1)])  # Expand std to match the shape of noise_prediction
        score = -noise_prediction / std
        return score
    return score_fn

def get_inverse_step_fn(discretisation):
    # Discretisation sequence is ordered from biggest time to smallest time
    map_t_to_negative_dt = {}
    steps = len(discretisation)
    for i in range(steps):
        if i <= steps - 2:
            map_t_to_negative_dt[discretisation[i]] = discretisation[i + 1] - discretisation[i]
        elif i == steps - 1:
            map_t_to_negative_dt[discretisation[i]] = map_t_to_negative_dt[discretisation[i - 1]]

    def inverse_step_fn(t):
        if t in map_t_to_negative_dt.keys():
            return map_t_to_negative_dt[t]
        else:
            closest_t_key = discretisation[np.argmin(np.abs(discretisation - t))]
            return map_t_to_negative_dt[closest_t_key]
    
    return inverse_step_fn

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

        if discretisation is not None:
            self.inverse_step_fn = get_inverse_step_fn(discretisation.cpu().numpy())

    @abc.abstractmethod
    def update_fn(self, x, y, t):
        """One update of the predictor.
        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.
        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
        super().__init__(sde, score_fn, probability_flow, discretisation)
        self.probability_flow = probability_flow

    def update_fn(self, x, y, t):
        dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) # dt = -(1-self.sde.sampling_eps) / self.rsde.N
        drift, diffusion = self.rsde.sde(x, y, t)
        x_mean = x + drift * dt
      
        if self.probability_flow:
            return x_mean, x_mean
        else:
            z = torch.randn_like(x)
            x = x_mean + diffusion[(...,) + (None,) * len(x.shape[1:])] * torch.sqrt(-dt) * z
            return x, x_mean

@contextmanager
def evaluation_mode(model):
    """Temporarily set the model to evaluation mode."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def generate_samples(y, sde, diffusion_model, steps, shape, device):
    with evaluation_mode(diffusion_model):
        score_fn = get_score_fn(sde, diffusion_model)

        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device).type(torch.float32)
            timesteps = torch.linspace(sde.T, sde.sampling_eps, steps + 1, device=device)
            predictor = EulerMaruyamaPredictor(sde, score_fn, probability_flow=False, discretisation=timesteps)

            for i in tqdm(range(steps)):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = predictor.update_fn(x, y, vec_t)

    return x_mean

def plot_samples(samples):
    """Create a scatter plot of the generated samples in 2D or 3D depending on the dimension."""
    if samples.shape[1] == 3:
        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Generated 3D Samples")
    elif samples.shape[1] == 2:
        # 2D plot
        fig, ax = plt.subplots()
        ax.scatter(samples[:, 0], samples[:, 1], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title("Generated 2D Samples")
    else:
        raise ValueError("Samples should be 2D or 3D only.")

    return fig, ax

def save_plot_to_tensorboard(writer, fig, tag, global_step):
    """Save the plot to TensorBoard."""
    fig.canvas.draw()
    
    # Convert plot to numpy array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Add the image to TensorBoard
    writer.add_image(tag, img, global_step=global_step, dataformats='HWC')
    
    # Close the plot
    plt.close(fig)

def plot_and_save_histogram_of_norms(samples, writer, steps):
    # Calculate the norms of each sample
    norms = torch.norm(samples, dim=1).cpu().numpy()
    
    # Compute mean and standard deviation of the norms
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Create a histogram of the norms
    fig, ax = plt.subplots()
    ax.hist(norms, bins=30, color='blue', alpha=0.7)
    ax.set_title(f'Histogram of the Norms (mean={mean_norm:.2f}, std={std_norm:.2f})')
    ax.set_xlabel('Norm')
    ax.set_ylabel('Frequency')
    
    # Save the histogram plot to TensorBoard
    save_plot_to_tensorboard(writer, fig, 'Histogram of Norms', steps)

def generation_callback(y, writer, sde, diffusion_model, steps, shape, device, epoch):
    #y is the condition. 
    samples = generate_samples(y, sde, diffusion_model, steps, shape, device)
    
    if samples.shape[1] in [2, 3]:
        # Plot the samples
        fig, ax = plot_samples(samples)
        
        # Save the plot to TensorBoard with epoch number in the tag
        save_plot_to_tensorboard(writer, fig, f'Generated Samples/Epoch {epoch + 1}', epoch)
    
    
    # Plot and save the histogram of norms
    plot_and_save_histogram_of_norms(samples, writer, epoch)
