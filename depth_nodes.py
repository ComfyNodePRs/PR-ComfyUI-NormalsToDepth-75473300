import torch
import numpy as np
from scipy.fft import dct, idct

# ==============================================================================
# HELPER FUNCTIONS & ALGORITHMS
# ==============================================================================

def get_gradients_from_normal(normal_map):
    """Convert RGB normal map to gradients (dz/dx, dz/dy)."""
    norm = normal_map * 2.0 - 1.0
    nx = norm[:, :, 0]
    ny = norm[:, :, 1]
    nz = norm[:, :, 2]

    # Handle tiny nz values
    eps = 1e-6
    nz = np.where(np.abs(nz) < eps, eps * np.sign(nz), nz)
    
    p = -nx / nz
    q = -ny / nz
    return p, q

def normalize_output(depth_map):
    """Normalize depth map to [0, 1] range."""
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map)
    return (depth_map - d_min) / (d_max - d_min)

def frankot_chellappa(p, q):
    """Project gradients into Fourier domain, integrate, and inverse transform."""
    rows, cols = p.shape
    fp = np.fft.fft2(p)
    fq = np.fft.fft2(q)
    
    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    
    numerator = -1j * (u * fp + v * fq)
    denominator = (u**2 + v**2) + 1e-12 
    
    fz = numerator / denominator
    fz[0, 0] = 0
    
    z = np.fft.ifft2(fz).real
    return z

def poisson_reconstruction(p, q):
    """Solves Poisson equation: Laplacian(Z) = div(Gradients) using DCT."""
    rows, cols = p.shape
    dx_p = np.gradient(p, axis=1)
    dy_q = np.gradient(q, axis=0)
    divergence = dx_p + dy_q
    
    dct_div = dct(dct(divergence, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    x_coords = np.arange(cols)
    y_coords = np.arange(rows)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    eigenvalues = (2 * np.cos(np.pi * xv / cols) - 2) + \
                  (2 * np.cos(np.pi * yv / rows) - 2)
    
    eigenvalues[0, 0] = 1.0 
    dct_z = dct_div / eigenvalues
    dct_z[0, 0] = 0.0 
    
    z = idct(idct(dct_z, axis=1, norm='ortho'), axis=0, norm='ortho')
    return z

# ==============================================================================
# COMFYUI NODES
# ==============================================================================

class NormalToDepthFrankotChellappa:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "invert_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "process"
    CATEGORY = "DepthSolvers"

    def process(self, normal_map, invert_output):
        nm_np = normal_map.cpu().numpy()
        batch_size, h, w, c = nm_np.shape
        output_batch = []

        for i in range(batch_size):
            p, q = get_gradients_from_normal(nm_np[i])
            z = frankot_chellappa(p, q)
            z_norm = normalize_output(z)
            if invert_output:
                z_norm = 1.0 - z_norm
            z_out = np.stack([z_norm, z_norm, z_norm], axis=-1)
            output_batch.append(z_out)

        return (torch.from_numpy(np.array(output_batch)).float(),)


class NormalToDepthPoisson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "invert_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "process"
    CATEGORY = "DepthSolvers"

    def process(self, normal_map, invert_output):
        nm_np = normal_map.cpu().numpy()
        batch_size, h, w, c = nm_np.shape
        output_batch = []

        for i in range(batch_size):
            p, q = get_gradients_from_normal(nm_np[i])
            z = poisson_reconstruction(p, q)
            z_norm = normalize_output(z)
            if invert_output:
                z_norm = 1.0 - z_norm
            z_out = np.stack([z_norm, z_norm, z_norm], axis=-1)
            output_batch.append(z_out)

        return (torch.from_numpy(np.array(output_batch)).float(),)


class DepthMapMathCombiner:
    """
    Combines two depth maps using a mathematical expression.
    Inputs: X (depth_map_1), Y (depth_map_2), a, b
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_map_x": ("IMAGE",),
                "depth_map_y": ("IMAGE",),
                "expression": ("STRING", {"default": "a * X + b * Y", "multiline": False}),
                "a": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0}),
                "b": ("FLOAT", {"default": 1.0, "step": 0.01, "min": -100.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_depth",)
    FUNCTION = "combine"
    CATEGORY = "DepthSolvers"

    def combine(self, depth_map_x, depth_map_y, expression, a, b):
        # Ensure dimensions match for broadcasting if possible, or raise error
        if depth_map_x.shape != depth_map_y.shape:
             # Basic safety: if shapes don't match, try to use X's shape
             # In a real scenario, you might want to resize Y to X here.
             # For now, we assume user provides matching dimensions.
             pass

        # Define the context for the evaluation
        # We use Torch math functions for speed
        math_context = {
            "X": depth_map_x,
            "Y": depth_map_y,
            "a": a,
            "b": b,
            "torch": torch,
            "log": torch.log,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "sin": torch.sin,
            "cos": torch.cos,
            "abs": torch.abs,
            "max": torch.max,
            "min": torch.min,
            "pow": torch.pow
        }

        try:
            # Evaluate the expression
            # Example: "a * X + b * torch.log(Y + 0.001)"
            result = eval(expression, {"__builtins__": {}}, math_context)
            
            # Handle result if it's a scalar (rare, but possible)
            if isinstance(result, (int, float)):
                result = torch.full_like(depth_map_x, result)
            
            # Clamp result to 0-1 to stay valid as an image (Optional, but recommended for depth)
            result = torch.clamp(result, 0.0, 1.0)
            
            return (result,)
            
        except Exception as e:
            print(f"[DepthMapMathCombiner] Error evaluating expression: {e}")
            # Return X as fallback in case of error
            return (depth_map_x,)

# ==============================================================================
# MAPPINGS
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "FrankotChellappa": NormalToDepthFrankotChellappa,
    "PoissonReconstruction": NormalToDepthPoisson,
    "DepthMathCombiner": DepthMapMathCombiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrankotChellappa": "Normal to Depth (Frankot-Chellappa)",
    "PoissonReconstruction": "Normal to Depth (Poisson)",
    "DepthMathCombiner": "Depth Map Math Combiner"
}