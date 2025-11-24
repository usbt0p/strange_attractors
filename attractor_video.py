import os
import logging
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from colors import colorInterpolate
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def generateFilename(dir, experiment_name, name, interpolation, 
                         width, height, maxiterations, extension="png"):
        if not os.path.exists(dirpath := os.path.join(dir, experiment_name)):
            os.makedirs(dirpath) # ignoring experiment_name if empty is intentional
        if experiment_name:
            name = f"{name}_{interpolation}_{width}x{height}_{maxiterations}"
        # check if file exists and if so find a new name
        suffix = 0
        while os.path.exists(os.path.join(
            dirpath, newname := f"{name}{"_" if str(name) else ""}{suffix}.{extension}")):
            suffix += 1
        return os.path.join(dirpath, newname)


def videoAttractor(
    name,
    xmin, xmax,
    ymin, ymax,
    x, y,
    video_width=1080,  # Video resolution width
    video_height=1080,  # Video resolution height
    background_color=None,
    dir="output",
    experiment_name="",
    cmap='magma',  # Colormap for histogram mode
    pad_size=0,
    video_duration=10.0,  # Duration in seconds
    fps=30,  # Frames per second
):
    """Create a video of an attractor being drawn point by point.
    """            

    # Calculate total frames and points per frame
    total_frames = int(video_duration * fps)
    total_points = len(x)
    points_per_frame = max(1, total_points // total_frames)
    print(f"Generating video with {total_frames} frames, {points_per_frame} points per frame.")
    
    # Normalize points to video coordinates taking pad_size into account
    int_pad = int(pad_size)
    eff_w = max(1, video_width - 2 * int_pad)
    eff_h = max(1, video_height - 2 * int_pad)

    xs = (((np.array(x) - xmin) / (xmax - xmin)) * eff_w + int_pad).astype(int)
    ys = (((np.array(y) - ymin) / (ymax - ymin)) * eff_h + int_pad).astype(int)
    
    # Clip coordinates to video bounds
    xs = np.clip(xs, 0, video_width - 1)
    ys = np.clip(ys, 0, video_height - 1)
    
    # # Get colormap colors
    colormap = plt.colormaps[cmap]
    
    # Generate video filename
    pth = generateFilename(dir, experiment_name, name, "hist", video_width, video_height, "", "mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pth, fourcc, fps, (video_width, video_height))
    
    # Set background color
    if background_color:
        assert len(background_color) == 3 and \
            all(0 <= c <= 255 for c in background_color), "Invalid background color"
        bg_color = background_color
    else:
        bg_color = tuple(int(c * 255) for c in colormap(0.0)[:3])
    
    # we'll track point density for coloring
    density_map = np.zeros((video_height, video_width), dtype=np.float32)

    for frame_idx in tqdm(range(total_frames), desc="Generating video frames"):
        # clear frame
        frame = np.full((video_height, video_width, 3), bg_color, dtype=np.uint8)

        # new indexes of points
        start_idx = frame_idx * points_per_frame
        end_idx = min(start_idx + points_per_frame, total_points)
        
        # use np.add.at to prevent loops
        curr_xs = xs[start_idx:end_idx]
        curr_ys = ys[start_idx:end_idx]
        np.add.at(density_map, (curr_ys, curr_xs), 1)
        
        if np.max(density_map) > 0:
            # apply log density and normalize
            log_density = np.log1p(density_map)
            max_log = np.max(log_density)
            norm_density = log_density / max_log
            
            # opencv color format
            colored_map = colormap(norm_density)[:, :, :3] 
            colored_map = (colored_map * 255).astype(np.uint8)
            colored_map = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
            
            # mask the only drawn pixels
            mask = density_map > 0
            frame[mask] = colored_map[mask]

        out.write(frame)

    out.release()
    logging.info(f"saved attractor video to ./{pth}")

def computeDensity(
    xmin, xmax,
    ymin, ymax,
    x, y,
    width=800,
    height=800,
    density_sigma=0,  # Gaussian smoothing for histogram mode
):
    """Compute the density histogram of the attractor points.
    """
     
    # Normalize points to [0, 1] range
    xs = (np.array(x) - xmin) / (xmax - xmin)
    ys = (np.array(y) - ymin) / (ymax - ymin)

    # Create a 2D histogram grid
    H2, _, _ = np.histogram2d(xs, ys, bins=[width, height], range=[[0, 1], [0, 1]])
    if np.sum(H2) == 0:
        logging.debug(f"Empty histogram for attractor")
        return

    density = H2
    if density_sigma > 0: density = gaussian_filter(H2, sigma=density_sigma)
    density[density < 1e-10] = 1e-10 # Avoid zero values 
    img_data = np.log1p(density) # log transform for better dynamic range
    # normalize again to [0, 1]
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) 
        
    return img_data.T[::-1]

if __name__ == "__main__":

    from lyapunov_exponents import loadAttractor, generateAttractorFromParameters
    
    RENDER_ITERATIONS = 10_000_000

    attractors = [15, 21, 22, 25, 29, 34]

    for n in attractors:
        print(f"Processing attractor {n}...")

        params = loadAttractor(f"/home/usbt0p/Programs/strange_attractors/set_num/out/{n}_0.json")
        x, y, xmin, xmax, ymin, ymax = \
            generateAttractorFromParameters(params, RENDER_ITERATIONS)
        
        print(f"Attractor {n} loaded and generated.")

        videoAttractor(
            f"density_{n}",
            xmin, xmax,
            ymin, ymax,
            x, y,
            video_duration=10.0,
            fps=20,
            video_width=1000,
            video_height=1000,
            pad_size=20,
            dir="set_num",
            cmap='magma'
        )

        print(f"Attractor {n} video generated.")
        print()