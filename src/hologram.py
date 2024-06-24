import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Converts amplitude and phase to rectangular coordinates
def polar_to_rect(amplitude, phase):
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    return real, imag

# Converts rectangular coordinates to amplitude and phase
def rect_to_polar(real, imag):
    amplitude = torch.sqrt(real**2 + imag**2)
    phase = torch.atan2(imag, real)
    return amplitude, phase

# Propagates a complex field using the Angular Spectrum Method (ASM)
"""
k: Wavenumber
fx , fy: Frequency coordinates
H: Transfer function for propagation
field_ftt: FFT
propagated_field: Inverse FFT

"""
def propagate_field(field, prop_dist, wavelength, feature_size):
    # Angular Spectrum Method (ASM) propagation
    k = 2 * np.pi / wavelength
    nx, ny = field.shape[-2:]
    fx = torch.fft.fftfreq(nx, d=feature_size, device=field.device)
    fy = torch.fft.fftfreq(ny, d=feature_size, device=field.device)
    fx, fy = torch.meshgrid(fx, fy, indexing='ij')
    H = torch.exp(1j * k * prop_dist * torch.sqrt(1 - (wavelength * fx)**2 - (wavelength * fy)**2))
    H = torch.fft.ifftshift(H)
    field_fft = torch.fft.fft2(field)
    propagated_field = torch.fft.ifft2(field_fft * H)
    return propagated_field

# Reduce the precision.granularity of the phase values to a specified number of levels
def quantize_phase(phase, levels):
    return torch.round(phase / (2 * np.pi / levels)) * (2 * np.pi / levels)

# After quantizing the phase values, this further refines the quantized values to reduce visible artifacts caused by the quantization process.
def error_diffusion(phase, levels):
    # Floyd-Steinberg Error Diffusion
    quantized = quantize_phase(phase, levels)
    error = phase - quantized
    error_padded = nn.functional.pad(error, (1, 1, 1, 1))
    height, width = phase.shape[-2:]
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            error_here = error_padded[..., y, x]
            error_padded[..., y, x+1] += error_here * 7 / 16
            error_padded[..., y+1, x-1] += error_here * 3 / 16
            error_padded[..., y+1, x] += error_here * 5 / 16
            error_padded[..., y+1, x+1] += error_here * 1 / 16
    return quantized + error_padded[..., 1:-1, 1:-1]

# Double Phase Hologram Simulation
def double_phase_hologram_simulation(init_phase, target_image, num_iters, prop_dist, wavelength, feature_size, quant_levels):
    # Converts target_image to torch tensor and adjust if needed to grayscale (already in)
    target_amp = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Initial complex field
    real, imag = polar_to_rect(torch.ones_like(init_phase), init_phase)
    slm_field = torch.complex(real, imag)

    for k in range(num_iters):
        # SLM/DMD to image plane
        recon_field = propagate_field(slm_field, prop_dist, wavelength, feature_size)
        
        # First phase manipulation (error diffusion)
        recon_phase_1 = quantize_phase(torch.angle(recon_field), quant_levels)
        recon_phase_1 = error_diffusion(recon_phase_1, quant_levels)
        
        # Second phase manipulation (additional error diffusion)
        recon_phase_2 = quantize_phase(recon_phase_1, quant_levels)
        recon_phase_2 = error_diffusion(recon_phase_2, quant_levels)
        
        # Image to SLM/MEM plane
        slm_field = propagate_field(torch.polar(torch.abs(recon_field), recon_phase_2), -prop_dist, wavelength, feature_size)
        slm_amp, slm_phase = rect_to_polar(slm_field.real, slm_field.imag)
        slm_phase = quantize_phase(slm_phase, quant_levels)
        slm_phase = error_diffusion(slm_phase, quant_levels)
        slm_field = torch.polar(torch.ones_like(slm_amp), slm_phase)
        
        # Prints progress
        if k % 10 == 0 or k == num_iters - 1:
            print(f"Iteration {k+1}/{num_iters} completed")

    return slm_field.angle()

"""
In each iteration it should:

Propagates the field to the image plane
Applies phase quantization and error diffusion
Propagates back to the SLM/MEMs plane and updates the phase.
See progress every 10 iteration

"""

# Parameters
wavelength = 532e-9  # 532 nm (green laser)
feature_size = 6.4e-6  # 6.4 micrometers
prop_dist = 0.1  # 10 cm
num_iters = 50  # Reduced for quicker testing
quant_levels = 256

# Load image
image_path = 'src/hologram.py'
image = Image.open(image_path).convert('L')  # Convert to grayscale if necessary
target_image = np.array(image)

# Resize the image for quicker testing to 100x100
target_image = np.array(image.resize((100, 100), Image.BILINEAR))

# Generate initial random phase
H, W = target_image.shape
init_phase = torch.rand(1, 1, H, W) * 2 * np.pi

# Runs the double phase hologram simulation to get the reconstructed phase
recon_phase = double_phase_hologram_simulation(init_phase, target_image, num_iters, prop_dist, wavelength, feature_size, quant_levels)

# Plots the target image and the reconstructed phase for visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Target Image")
plt.imshow(target_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Reconstructed Phase")
plt.imshow(recon_phase[0, 0].cpu().numpy(), cmap='gray')
plt.show()
