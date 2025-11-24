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
    use_density=True,
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
    # colormap = plt.colormaps[cmap]
    
    # Generate video filename
    pth = generateFilename(dir, experiment_name, name, "hist", video_width, video_height, "", "mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pth, fourcc, fps, (video_width, video_height))
    
    # # Set background color
    # if background_color:
    #     bg_color = background_color
    # else:
    #     bg_color = tuple(int(c * 255) for c in colormap(0.0)[:3])
    
    # Track point density for coloring
    # density_map = np.zeros((video_height, video_width), dtype=np.float32)
    # # Track which pixels have been visited
    # visited_pixels = set()

    bg_color = (255, 255, 255)  # White background in BGR

    # Create fresh frame with background
    frame = np.full((video_height, video_width, 3), bg_color, dtype=np.uint8)
    
    for frame_idx in tqdm(range(total_frames), desc="Generating video frames"):
        
        # Calculate which points to add this frame
        start_idx = frame_idx * points_per_frame
        end_idx = min(start_idx + points_per_frame, total_points)
        
        # Add new points to density map
        # for i in range(start_idx, end_idx):
        #     x_coord, y_coord = xs[i], ys[i]
        #     density_map[y_coord, x_coord] += 1
        #     visited_pixels.add((x_coord, y_coord))
        
        # Redraw all visited pixels with updated colors
        # if len(visited_pixels) > 0:
        #     max_density = np.max(density_map)
            
        #     for x_coord, y_coord in visited_pixels:
        #         density_value = density_map[y_coord, x_coord]
                
        #         # Get color based on density or iteration
        #         if use_density:
        #             # Use density for coloring
        #             color_value = min(density_value / max_density, 1.0) if max_density > 0 else 0
        #         else:
        #             # For non-density mode, use the last iteration that hit this pixel
        #             # Find the last occurrence of this coordinate
        #             last_occurrence = 0
        #             for i in range(end_idx - 1, -1, -1):
        #                 if xs[i] == x_coord and ys[i] == y_coord:
        #                     last_occurrence = i
        #                     break
        #             color_value = last_occurrence / total_points
                
        #         # Get RGB color from colormap
        #         rgb = colormap(color_value)[:3]
        #         bgr_color = tuple(int(c * 255) for c in rgb[::-1])  # Convert RGB to BGR for OpenCV
            
        for i in range(start_idx, end_idx):
            cv2.circle(frame, (xs[i], ys[i]), 
                       0, (0, 0, 0), -1)  # Draw in black
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    logging.info(f"saved attractor video to ./{pth}")

def computePoints(
    xmin, xmax,
    ymin, ymax,
    x, y,
    width=800,
    height=800,
    density_sigma=0,  # Gaussian smoothing for histogram mode
):
     
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
    density[density < 1e-10] = 1e-10 # Avoid zero values before log transform
    img_data = np.log1p(density) # Apply log transform for better dynamic range
    img_data = (img_data - np.min(img_data)) / ( # normalize again to [0, 1]
        np.max(img_data) - np.min(img_data))

    return img_data.T[::-1]

if __name__ == "__main__":


    from lyapunov_exponents import loadAttractor, generateAttractorFromParameters
    
    RENDER_ITERATIONS = 10_000_000

    attractors = [15, 21] #, 22, 25, 29, 34]

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
            video_duration=7.0,
            fps=10,
            video_width=1000,
            video_height=1000,
            pad_size=20,
            dir="set_num",
            use_density=True,
            cmap='magma'
        )

        print(f"Attractor {n} video generated.")
        print()