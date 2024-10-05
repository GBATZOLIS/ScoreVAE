import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
from contextlib import contextmanager
import abc
import torchvision.utils as vutils
import math
from torch.nn import DataParallel
import torch.multiprocessing as mp
import ctypes  # Add this line
import os 
import io
from PIL import Image

# Set the Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

def unflatten_structure(flattened_structure, num_atoms=4):
    """
    Reshapes the structure back from (num_residues*num_atoms, 3) to (num_residues, num_atoms, 3).
    """
    total_atoms = flattened_structure.shape[0]
    assert total_atoms % num_atoms == 0, "The total number of atoms should be divisible by num_atoms."
    
    num_residues = total_atoms // num_atoms
    unflattened_structure = flattened_structure.reshape(num_residues, num_atoms, 3)
    return unflattened_structure

def plot_to_tensorboard(writer, fig, tag, epoch, angle_idx):
    """
    Convert a matplotlib figure to a NumPy array and write it to TensorBoard.
    """
    # Save the plot to a bytes buffer in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert the bytes buffer to a PIL image
    img = Image.open(buf)
    img = np.array(img)

    # Add the image to TensorBoard
    writer.add_image(f'{tag}_angle_{angle_idx}', img.transpose(2, 0, 1), epoch)  # Transpose to (C, H, W)

def save_protein_plot_to_tensorboard(atom_positions, backbone_mask, angles, writer, tag, epoch):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot backbone atom positions (blue points)
    for i in range(atom_positions.shape[0]):
        for j in range(atom_positions.shape[1]):
            if backbone_mask[i, j] > 0:
                ax.scatter(atom_positions[i, j, 0], atom_positions[i, j, 1], atom_positions[i, j, 2], c='b')

    # Save the plot from different angles
    for i, angle in enumerate(angles):
        ax.view_init(elev=angle[0], azim=angle[1])
        plot_to_tensorboard(writer, fig, tag, epoch, i)

    plt.close(fig)

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

class DDIMPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    #assert isinstance(sde, sde_lib.VPSDE), 'ddim sampler is supported only for the VPSDE currently.'

  def update_fn(self, z_t, y, t):
    #compute the negative timestep
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N 
    s = t + dt
    #compute the coefficients
    a_t, sigma_t = self.sde.perturbation_coefficients(t[0])
    a_s, sigma_s = self.sde.perturbation_coefficients(s[0])

    denoising_value = (sigma_t**2 * self.score_fn(z_t, y, t) + z_t)/a_t
    z_s = sigma_s/sigma_t * z_t + (a_s - sigma_s/sigma_t*a_t)*denoising_value
    return z_s, z_s

class HeunPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    #we implement the PECE method here. This should give us quadratic order of accuracy.

  def f(self, x, y, t):
      drift, diffusion = self.sde.sde(x, t)
      score = self.score_fn(x, y, t)
      drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * 0.5
      return drift

  def predict(self, x, f_0, h):
    prediction = x + f_0 * h
    return prediction
  
  def correct(self, x, f_1, f_0, h):
    correction = x + h/2 * (f_1 + f_0)
    return correction

  def update_fn(self, x, y, t):
      h = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-(1-self.sde.sampling_eps) / self.rsde.N

      #evaluate
      f_0 = self.f(x, y, t)
      #predict
      x_1 = self.predict(x, f_0, h)
      #evaluate
      f_1 = self.f(x_1, y, t+h)
      #correct once
      x_2 = self.correct(x, f_1, f_0, h)

      return x_2, x_2
  
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
    diffusion_model.to(device)  # Ensure the model is on the correct device
    with evaluation_mode(diffusion_model):
        score_fn = get_score_fn(sde, diffusion_model)
        with torch.no_grad():
            x_mean = Algorithm1(sde, steps, score_fn, shape, device, y)

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
    
    if len(samples.shape[1:]) == 1: #euclidean data, i.e. shape=(batchsize, ambient_dim)
        if samples.shape[1] in [2, 3]:
            # Plot the samples
            fig, ax = plot_samples(samples)
            
            # Save the plot to TensorBoard with epoch number in the tag
            save_plot_to_tensorboard(writer, fig, f'Generated Samples/Epoch {epoch + 1}', epoch)
        
        
        # Plot and save the histogram of norms
        plot_and_save_histogram_of_norms(samples, writer, epoch)
    elif len(samples.shape[1:]) == 3: #assume images of shape (channels, width, height)
        # samples is a batch of images of shape (batchsize, c, w, h)
        num_rows = int(math.sqrt(samples.shape[0]))
        
        # Create a grid of images
        grid = vutils.make_grid(samples, nrow=num_rows, normalize=True, scale_each=True)
        
        # Save the image grid to TensorBoard
        writer.add_image(f'Generated Images', grid, epoch)
    elif len(samples.shape[1:]) == 2: #assumes proteins of shape (batchsize, num_aminoacids*num_atoms, 3)
        num_aminoacids = 256  # Adjust based on config.data.max_seq_length
        num_atoms = 4  # Adjust based on config.data.num_backbone_atoms

        # Reshape the protein batch (batchsize, num_aminoacids * num_atoms, 3) to (batchsize, num_aminoacids, num_atoms, 3)
        samples_reshaped = samples.view(samples.shape[0], num_aminoacids, num_atoms, 3).detach().cpu().numpy()

        # Define the angles for the 2D projections
        angles = [(30, 30), (90, 0), (0, 90), (60, 60)]

        # Visualize the first four proteins from four different angles
        for i in range(4):
            atom_positions = samples_reshaped[i]  # First protein in batch
            # Generate a mask (since we don't have an explicit one, assume all atoms are valid)
            backbone_mask = np.ones((num_aminoacids, num_atoms))

            # Save the protein from multiple angles to TensorBoard
            save_protein_plot_to_tensorboard(atom_positions, backbone_mask, angles, writer, f"protein_{i}", epoch)

def generate_specified_num_samples(num_samples, sde, diffusion_model, steps, shape, device):
    with evaluation_mode(diffusion_model):
        score_fn = get_score_fn(sde, diffusion_model)
        with torch.no_grad():
            all_samples = []
            num_iterations = (num_samples + shape[0] - 1) // shape[0]  # Calculate the number of iterations required
            for _ in tqdm(range(num_iterations), desc="Generating samples for FID evaluation"):
                x_mean = Algorithm1(sde, steps, score_fn, shape, device, y=None)
                all_samples.append(x_mean.cpu())  # Move to CPU immediately after generation

            all_samples = torch.cat(all_samples, dim=0)
            all_samples = all_samples[:num_samples]  # Discard redundant samples to get exactly `num_samples`

    return all_samples

def generate_samples_on_device(device_id, num_samples_per_device, sde, diffusion_model, steps, shape):
    device = torch.device(f'cuda:{device_id}')
    diffusion_model.to(device)
    
    with evaluation_mode(diffusion_model):
        score_fn = get_score_fn(sde, diffusion_model)

        with torch.no_grad():  # Ensure no_grad is used to prevent memory leaks
            all_samples = []
            samples_per_batch = shape[0]
            num_iterations = math.ceil(num_samples_per_device / samples_per_batch)  # Calculate the number of iterations required

            for _ in tqdm(range(num_iterations), desc=f"Generating samples for FID evaluation on {device}"):
                x_mean = Algorithm1(sde, steps, score_fn, shape, device, y=None)
                all_samples.append(x_mean.cpu())  # Move to CPU immediately after generation

            all_samples = torch.cat(all_samples, dim=0)
            all_samples = all_samples[:num_samples_per_device]  # Discard redundant samples to get exactly `num_samples_per_device`
    
    return all_samples

def generate_specified_num_samples_parallel(num_samples, sde, diffusion_model, steps, shape, device_ids):
    num_gpus = len(device_ids)
    samples_per_gpu = math.ceil(num_samples / num_gpus)

    with mp.Pool(processes=num_gpus) as pool:
        results = [pool.apply_async(generate_samples_on_device, args=(device_id, samples_per_gpu, sde, diffusion_model, steps, shape)) for device_id in device_ids]
        all_samples = [res.get() for res in results]

    # Print the first element of the first tensor
    print(all_samples[0][0,0,::])

    all_samples = torch.cat(all_samples, dim=0)
    return all_samples[:num_samples]

def Algorithm1(sde, steps, score_fn, shape, device, y=None):
    def get_ode_derivative(sde, score_fn):
        def ode_derivative(x, y, t):
            t = torch.ones(x.size(0), device=t.device) * t
            drift, diffusion = sde.sde(x, t)
            score = score_fn(x, y, t)
            drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * 0.5
            return drift
        return ode_derivative

    #diffusion_times = fullrange_infer_timesteps(sde, steps, rho=7) #timesteps according to Karras sigma parametrisation. We use rho=7. FID=3.69 for cifar10 - range = [1e-3, 1]
    diffusion_times = karras_preferred_timesteps(sde, steps, rho=7) #edm_sigma = [0.002, 80].
    #diffusion_times = torch.cat((torch.linspace(sde.T, sde.sampling_eps, steps), torch.tensor([0.0]))) #linear timesteps

    diffusion_times = diffusion_times.to(device)
    ode_derivative = get_ode_derivative(sde, score_fn)
    
    x = sde.prior_sampling(shape).to(device).type(torch.float32)

    for i in range(steps):
        t_i = diffusion_times[i] #get the diffusion time 
        t_iplusone = diffusion_times[i+1] #get the target/next diffusion time
        h = t_iplusone - t_i #get the diffusion timestep, t_{i+1} - t_i
        d_i = ode_derivative(x, y, t_i) #evaluate the derivative of the ODE at t_i
        x_iplusone = x + h * d_i # euler step from t_i to t_{i+1}

        if i < steps - 1: #apply second order correction in all but the last step
            d_iplusone = ode_derivative(x_iplusone, y, t_iplusone) #evaluate the derivate at t_iplusone
            x = x + 0.5 * h * (d_i + d_iplusone) #explicit trapezoidal rule at t_iplusone
        else:
            x = x_iplusone
    
    return x #return noise-free sample

def fullrange_infer_timesteps(sde, num_steps, rho):
    def get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, integration_steps):
        def parametrized_edm_sigma_fn(step):
            return (edm_sigma_max**(1/rho) + step/(integration_steps-1) * (edm_sigma_min**(1/rho) - edm_sigma_max**(1/rho)))**rho
        return parametrized_edm_sigma_fn

    def get_inverse_edm_sigma_fn(sde):
        beta_d = sde.beta_1 - sde.beta_0
        beta_0 = sde.beta_0
        def inverse_edm_sigma_fn(edm_sigma_t):
            # Compute coefficients
            a = 0.5 * beta_d
            b = beta_0
            c = -torch.log(edm_sigma_t**2 + 1)
            # Compute the discriminant
            discriminant = b**2 - 4 * a * c
            # Compute the positive root of the quadratic equation
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
            return t
        
        return inverse_edm_sigma_fn
    
    tmin = torch.tensor(sde.sampling_eps)
    edm_sigma_min = sde.edm_coefficients(tmin)[1]
    #print(f'edm_sigma_min: {edm_sigma_min}')
    tmax = torch.tensor(sde.T)
    edm_sigma_max = sde.edm_coefficients(tmax)[1]
    #print(f'edm_sigma_max: {edm_sigma_max}')

    steps = torch.arange(num_steps)
    parametrized_edm_sigma_fn = get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, num_steps)
    parametrized_edm_sigmas = parametrized_edm_sigma_fn(steps)
    inverse_edm_sigma_fn = get_inverse_edm_sigma_fn(sde)
    diffusion_times = inverse_edm_sigma_fn(parametrized_edm_sigmas)
    diffusion_times = torch.cat((diffusion_times, torch.tensor([0.0])))
    #print(diffusion_times)
    return diffusion_times

def karras_preferred_timesteps(sde, num_steps, rho):
    def get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, integration_steps):
        def parametrized_edm_sigma_fn(step):
            return (edm_sigma_max**(1/rho) + step/(integration_steps-1) * (edm_sigma_min**(1/rho) - edm_sigma_max**(1/rho)))**rho
        return parametrized_edm_sigma_fn
    
    def get_inverse_edm_sigma_fn(sde):
        beta_d = sde.beta_1 - sde.beta_0
        beta_0 = sde.beta_0
        
        def inverse_edm_sigma_fn(edm_sigma_t):
            # Compute coefficients
            a = 0.5 * beta_d
            b = beta_0
            c = -torch.log(edm_sigma_t**2 + 1)

            # Compute the discriminant
            discriminant = b**2 - 4 * a * c

            # Compute the positive root of the quadratic equation
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
            
            return t
        
        return inverse_edm_sigma_fn
    
    inverse_edm_sigma_fn = get_inverse_edm_sigma_fn(sde)

    #C.1.4 Karras paper.
    edm_sigma_min = torch.tensor(0.002)
    time_edm_sigma_min = inverse_edm_sigma_fn(edm_sigma_min)
    print(f'Min diffusion time: {time_edm_sigma_min.item()}')
    edm_sigma_max = torch.tensor(80)
    time_edm_sigma_max = inverse_edm_sigma_fn(edm_sigma_max)
    print(f'Max diffusion time: {time_edm_sigma_max.item()}')

    steps = torch.arange(num_steps)
    parametrized_edm_sigma_fn = get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, num_steps)
    parametrized_edm_sigmas = parametrized_edm_sigma_fn(steps)
    diffusion_times = inverse_edm_sigma_fn(parametrized_edm_sigmas)
    diffusion_times = torch.cat((diffusion_times, torch.tensor([0.0])))
    return diffusion_times
    
def inspect_timesteps(sde, num_steps, eval_dir):
    def get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, integration_steps):
        def parametrized_edm_sigma_fn(step):
            return (edm_sigma_max**(1/rho) + step/(integration_steps-1) * (edm_sigma_min**(1/rho) - edm_sigma_max**(1/rho)))**rho
        return parametrized_edm_sigma_fn

    def get_inverse_edm_sigma_fn(sde):
        beta_d = sde.beta_1 - sde.beta_0
        beta_0 = sde.beta_0
        
        def inverse_edm_sigma_fn(edm_sigma_t):
            # Compute coefficients
            a = 0.5 * beta_d
            b = beta_0
            c = -torch.log(edm_sigma_t**2 + 1)

            # Compute the discriminant
            discriminant = b**2 - 4 * a * c

            # Compute the positive root of the quadratic equation
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
            
            return t
        
        return inverse_edm_sigma_fn

    inverse_edm_sigma_fn = get_inverse_edm_sigma_fn(sde)

    #C.1.4 Karras paper.
    edm_sigma_min = torch.tensor(0.002)
    time_edm_sigma_min = inverse_edm_sigma_fn(edm_sigma_min)
    print(f'Min diffusion time: {time_edm_sigma_min.item()}')
    edm_sigma_max = torch.tensor(80)
    time_edm_sigma_max = inverse_edm_sigma_fn(edm_sigma_max)
    print(f'Max diffusion time: {time_edm_sigma_max.item()}')

    timesteps = torch.linspace(time_edm_sigma_max, time_edm_sigma_min, num_steps)
    steps = torch.arange(num_steps)

    edm_sigmas = sde.edm_coefficients(timesteps)[1]

    # Plot EDM Sigmas
    plt.figure()
    plt.plot(steps, edm_sigmas, label='normal sigmas')

    for rho in [1, 3, 7]:
        parametrized_edm_sigma_fn = get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, num_steps)
        parametrized_edm_sigmas = parametrized_edm_sigma_fn(steps)
        
        plt.plot(steps, parametrized_edm_sigmas, label=f'rho={rho}')

    plt.xlabel('Timesteps')
    plt.ylabel('EDM Sigmas')
    plt.title('EDM Sigmas over Timesteps')
    plt.legend()

    plot_path = os.path.join(eval_dir, 'edm_sigmas_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Additional plots for diffusion times and SNR vs. integration steps
    normal_diffusion_times = inverse_edm_sigma_fn(edm_sigmas)

    plt.figure(figsize=(14, 6))

    # Diffusion Times Plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, normal_diffusion_times, label='normal sigmas')

    for rho in [1, 3, 7]:
        parametrized_edm_sigma_fn = get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, num_steps)
        parametrized_edm_sigmas = parametrized_edm_sigma_fn(steps)
        diffusion_times = inverse_edm_sigma_fn(parametrized_edm_sigmas)
        
        plt.plot(steps, diffusion_times, label=f'rho={rho}')

    plt.xlabel('Integration Steps')
    plt.ylabel('Diffusion Times')
    plt.title('Diffusion Times over Integration Steps')
    plt.legend()

    # SNR Plot
    plt.subplot(1, 2, 2)
    snr_values = sde.snr(normal_diffusion_times)
    plt.plot(steps, snr_values, label='normal sigmas')

    for rho in [1, 3, 7]:
        parametrized_edm_sigma_fn = get_parametrized_edm_sigma_fn(rho, edm_sigma_min, edm_sigma_max, num_steps)
        parametrized_edm_sigmas = parametrized_edm_sigma_fn(steps)
        diffusion_times = inverse_edm_sigma_fn(parametrized_edm_sigmas)
        snr_values = sde.snr(diffusion_times)
        
        plt.plot(steps, snr_values, label=f'rho={rho}')

    plt.xlabel('Integration Steps')
    plt.ylabel('SNR')
    plt.title('SNR over Integration Steps')
    plt.yscale('log')
    plt.legend()

    plot_path = os.path.join(eval_dir, 'diffusion_times_and_snr_plot.png')
    plt.savefig(plot_path)
    plt.close()