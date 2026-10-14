import os
import json
import glob
import logging

import numba
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from config import RenderConfig, CreateConfig
from equations import Equation

@numba.njit(fastmath=True)
def compute_lyapunov(step, a, b, iterations, discard=1000):
    """Compute the Lyapunov exponent of the attractor defined by step and coefficients a and b.

    Args:
        step: jitted map function (prev_x, prev_y, a, b) -> (x, y)
        a, b: arrays of coefficients for the x and y equations
        iterations: iterations to use when calculating the attractor
        discard: number of initial iterations to discard for transient behavior.
    """

    # This is the initial point, it will serve as the seed for the series
    # Even though the series is chaotic, it is deterministic
    x_val = np.random.uniform(-0.5, 0.5)
    y_val = np.random.uniform(-0.5, 0.5)

    x, y = [x_val], [y_val]

    # min and max bounds
    xmin = 1e32
    ymin = 1e32
    xmax = -1e32
    ymax = -1e32
    lyapunov = 0.0

    # calculate the initial separation between two nearby points and ensure it's not zero
    d0 = -1.0
    while d0 <= 0: # until the points are not identical
        # xe and ye are coords of our initial point, but slightly perturbed
        xe = x_val + np.random.uniform(-0.5, 0.5) / 1000.0
        ye = y_val + np.random.uniform(-0.5, 0.5) / 1000.0
        # dx and dy aid us in later filtering point or infinite attractors,
        # they are the vector between the seed and the perturbed point
        dx = x_val - xe
        dy = y_val - ye
        # this is just the distance of the previous vector,
        # a theoretical infinitesimal that will be used to renormalize the
        # distance between the two points after each iteration
        d0 = math.sqrt(dx * dx + dy * dy) # (epsilon in Bennetin method)

    drawit = True

    # keep previous value to avoid acessing list
    prev_x = x_val
    prev_y = y_val

    # Calculate the attractor
    for i in range(1, iterations): # start at 1 to avoid using x[-1]

        x_i, y_i = step(prev_x, prev_y, a, b)

        x.append(x_i)
        y.append(y_i)

        # this represents a nearby point, but slightly separated, on another orbit
        # we'll use this to calculate the lyapunov exponent
        xenew, yenew = step(xe, ye, a, b)

        # Update the bounds
        # if is faster than min/max functions
        if x_i < xmin: xmin = x_i
        if x_i > xmax: xmax = x_i
        if y_i < ymin: ymin = y_i
        if y_i > ymax: ymax = y_i

        # Does the series tend to infinity
        if xmin < -1e10 or ymin < -1e10 or xmax > 1e10 or ymax > 1e10:
            drawit = False
            #logging.info("infinite attractor")
            break

        # if the vector between current and prev point is too small,
        # the series will tend to a point
        if abs(x_i - prev_x) < 1e-10 and abs(y_i - prev_y) < 1e-10:
            drawit = False
            #logging.info("point attractor")
            break

        # Calculate the lyapunov exponents using
        # we ignore the first iters to allow the series to settle down into a pattern
        if i > discard:
            # this approach of approximating the lyapunov exponent is called the Bennetin method
            # it's not 100% equal to the mathematical definition, but it's a good approximation

            # two vectors are computed as the difference between the current point and
            # our perturbed point (xenew, yenew)
            dx = x_i - xenew
            dy = y_i - yenew

            dd_sq = dx * dx + dy * dy
            # prevent collapse due to identical points
            if dd_sq < 1e-30:
                drawit = False
                break

            dd = math.sqrt(dd_sq) # compute the distance between the norm

            # add the log of the ratio between the two current point's distances and the initial infinitesimal distance
            lyapunov += math.log(dd / d0)

            # renormalize the perturbed point to be d0 away from the current point (to avoid overflow or underflow of the value)
            scaler = d0 / dd
            xe = x_i + dx * scaler
            ye = y_i + dy * scaler
        else:
            # Actualizamos la sombra aunque no calculemos Lyapunov todavía
            xe = xenew
            ye = yenew

        # Actualizar previo para la siguiente vuelta
        prev_x = x_i
        prev_y = y_i

        # TODO check for correctness of lyapunov normalization
        # lyapunov /= (MAXITERATIONS - 1000) # normalize
        # and what about T?

    return lyapunov, drawit, x, y, a, b, xmin, xmax, ymin, ymax

def createAttractors(create_cfg: CreateConfig):
    '''Create strange attractors and save them as images and JSON files.
    If you want more insights into the algorithm:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7512692/#sec3-entropy-20-00175
    https://paulbourke.net/fractals/lyapunov/
    '''

    out_path = create_cfg.out_path
    examples = create_cfg.examples
    iterations = create_cfg.iterations
    equation = create_cfg.equation

    assert out_path is not None and out_path != "", "Output path must be specified"

    # by default create output directory and attr_number.txt if not exists
    if not os.path.exists(f"{out_path}/attr_number.txt"):
        print(f"Creating output directory {out_path}")
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/attr_number.txt", "x") as f:
            f.write("0")

    with open(f"{out_path}/attr_number.txt", "r") as f:
        file_index = int(f.read().strip())

    count = 0
    pbar = tqdm(total=examples, desc="Generating attractors")

    try: # catch keyboard interrupt to save progress
        while count < examples:

            a, b = equation.random_coefs()
            lyapunov, drawit, x, y, ax, ay, xmin, xmax, ymin, ymax = compute_lyapunov(
                equation.step, a, b, iterations, 1000,
            )

            # Classify the series according to lyapunov
            if drawit:
                if abs(lyapunov) < 10:
                    logging.info("neutrally stable")
                    drawit = False
                elif lyapunov < 0:
                    logging.info(f"periodic {lyapunov} ")
                    drawit = False
                else:
                    logging.info(f"chaotic {lyapunov} ")

            lyap_payload = {"lyapunov": lyapunov,
                            "normalized_lyapunov": lyapunov / (iterations - 1000)}
            # Save the image
            if drawit:
                saveAttractor(
                    file_index,
                    out_path,
                    equation.name,
                    ax, ay,
                    xmin, xmax,
                    ymin, ymax,
                    x[0], y[0],
                    lyap_payload,
                )
                drawAttractor(
                    file_index,
                    xmin, xmax,
                    ymin, ymax,
                    x, y,
                    render_cfg=RenderConfig(
                        dir=out_path,
                        iterations=iterations,
                        width=500,
                        height=500,
                        interpolation=None,
                    ),
                    background_color=(255, 255, 255),
                )
                file_index += 1
                count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("Interrupted by user, saving progress...")
    finally:
        # save the current file index
        with open(f"{out_path}/attr_number.txt", "w") as f:
            f.write(str(file_index))
        pbar.close()


def saveAttractor(name, path, equation, a, b, xmin, xmax, ymin, ymax, x_o, y_o, lyapunov):
    # Save the parameters in JSON format
    parameters = {
    # TODO consider adding bounds to the metadata
        "equation": equation,
        "bounds": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        },
        "coefficients": {
            "a": [a[i] for i in range(len(a))],
            "b": [b[i] for i in range(len(b))]
        },
        "seed_point": {
            "x": x_o,
            "y": y_o
        },
        "lyapunov": lyapunov
    }

    suffix = 0
    while os.path.exists(os.path.join(
        path, newname := f"{name}{'_' if str(name) else ''}{suffix}.json")):
        suffix += 1
    with open(os.path.join(path, newname), "w") as f:
        json.dump(parameters, f, indent=4)

def loadAttractor(pathname) -> dict:
    '''Load and return params with the JSON format in saveAttractor.'''
    with open(f"{pathname}", "r") as f:
        parameters = json.load(f)
    return parameters

@numba.njit(fastmath=True)
def compute_attractor_core(step, a, b, x0, y0, iters):
    """Core function to compute the attractor points using Numba for speed.
    step is the jitted map, a and b are arrays of coefficients,
    x0 and y0 are the seed points.
    Returns the computed x and y arrays containing the attractor points.
    """
    # preallocate memory
    x = np.zeros(iters)
    y = np.zeros(iters)
    x[0] = x0
    y[0] = y0

    for i in range(1, iters):
        x[i], y[i] = step(x[i-1], y[i-1], a, b)
    return x, y

def generateAttractorFromParameters(params : dict, iters : int):
    '''Interface to generate attractor points from loaded parameters.
    Returns the computed x and y arrays along with the bounds of the attractor.
    '''
    equation = Equation(params["equation"])
    # TODO consider adding bounds to the metadata

    # ensure numpy for numba
    a = np.array(params["coefficients"]["a"], dtype=np.float64)
    b = np.array(params["coefficients"]["b"], dtype=np.float64)

    xmin = params["bounds"]["xmin"]
    xmax = params["bounds"]["xmax"]
    ymin = params["bounds"]["ymin"]
    ymax = params["bounds"]["ymax"]
    x0 = params["seed_point"]["x"]
    y0 = params["seed_point"]["y"]

    x, y = compute_attractor_core(equation.step, a, b, x0, y0, iters)

    logging.info("Attractor generated from parameters.")
    return x, y, xmin, xmax, ymin, ymax

@numba.njit(fastmath=True, parallel=False)
def fast_histogram(x, y, xmin, xmax, ymin, ymax, width, height):
    """
    Calculate 2D histogram directly without creating intermediate arrays
    or using np.histogram2d.
    """

    grid = np.zeros((width, height), dtype=np.float32)
    n_points = len(x)

    scale_x = width / (xmax - xmin)
    scale_y = height / (ymax - ymin)

    for i in numba.prange(n_points):

        px = int((x[i] - xmin) * scale_x)
        py = int((y[i] - ymin) * scale_y)

        if px >= width: px = width - 1
        elif px < 0: px = 0

        if py >= height: py = height - 1
        elif py < 0: py = 0

        # 3. Acumular
        grid[px, py] += 1.0

    return grid

def computeDensity(
    xmin, xmax,
    ymin, ymax,
    x, y,
    width, height,
    density_sigma=0,
):
    """Compute the log-density histogram of the attractor points."""

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    H2 = fast_histogram(x_arr, y_arr, xmin, xmax, ymin, ymax, width, height)

    if not H2.any():
        logging.debug(f"Empty histogram for attractor")
        return None

    density = H2

    if density_sigma > 0:
        density = gaussian_filter(H2, sigma=density_sigma)

    density = np.maximum(density, 1e-10)

    img_data = np.log1p(density) # Log transform

    min_val = np.min(img_data)
    max_val = np.max(img_data)

    diff = max_val - min_val
    if diff > 0:
        img_data = (img_data - min_val) / diff
    else:
        img_data = np.zeros_like(img_data)

    # "rotate" the histogram to match coords
    return img_data.T[::-1]

def drawAttractor(
    name,
    xmin, xmax,
    ymin, ymax,
    x, y,

    render_cfg : RenderConfig,
    filter_threshold=0.02,

    # TODO tbd. what to do with this.
    cmap=None,  # Colormap for histogram mode
    density_sigma=0,  # Gaussian smoothing for histogram mode
    experiment_name="",

    start_color=(0, 0, 0),
    end_color=(0, 0, 0),
    background_color=None,
):
    """Create an image of the attractor using the provided parameters.
    Interplation is the function used to assign colors to points. The current best option
    is 'histogram', which is a point density based colorint, but other options based on "timestep"
    interpolation are available.
    Current quirks:
    - if the image already exists, a new name is generated by appending _0, _1, etc
    - if experiment_name is provided, it is used as a subdirectory and prefix for the filename
    - histogram mode ignores start and end colors, will always use bicubic interpolation
    - padding only works in histogram mode
    """

    if render_cfg.interpolation == 'histogram':
        assert cmap is not None, "At least one colormap must be specified for histogram interpolation"

    def generateFilename(dir, experiment_name, name, interpolation,
                         width, height, maxiterations, extension="png"):
        if not os.path.exists(dirpath := os.path.join(dir, experiment_name)):
            os.makedirs(dirpath, exist_ok=True) # ignoring experiment_name if empty is intentional
        if experiment_name:
            name = f"{name}_{interpolation}_{width}x{height}_{maxiterations}"
        # check if file exists and if so find a new name
        suffix = 0
        while os.path.exists(os.path.join(
            dirpath, newname := f"{name}{'_' if str(name) else ''}{suffix}.{extension}")):
            suffix += 1
        return os.path.join(dirpath, newname)

    W, H = render_cfg.width, render_cfg.height

    if render_cfg.interpolation == 'histogram':

        img_data = computeDensity(
            xmin, xmax,
            ymin, ymax,
            x, y,
            W, H,
            density_sigma,
        )

        # Create the plot
        plt.figure(figsize=(W / 100, H / 100), dpi=100)
        plt.imshow(
            img_data,
            origin="lower",
            interpolation="bicubic",
            cmap=cmap,
        )
        plt.axis("off")
        plt.tight_layout()

        # Save the image
        pth = generateFilename(render_cfg.dir, experiment_name, name, render_cfg.interpolation,
                               W, H, render_cfg.iterations)
        # TODO for alpha, something like np.array((*color, 1.0))??
        background = tuple(c / 255 for c in background_color) if background_color else tuple(cmap(0)[:3])
        plt.savefig(pth, bbox_inches="tight", pad_inches=render_cfg.pad_size, facecolor=background)
        plt.close()
        logging.info(f"saved attractor to ./{pth}")

    else:
        # this is mostly deprecated as histogram method works much better, only for creating the initial images
        from colors import colorInterpolate
        if background_color:
            img = Image.new("RGB", (W, H), background_color)
        else:
            img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        # iterate through all points and draw them with the color interpolated
        n_points = min(len(x), render_cfg.iterations)
        for i in range(1, n_points):
            # Map the point to image coordinates
            ix = int((x[i] - xmin) / (xmax - xmin) * (W - 1))
            iy = int((y[i] - ymin) / (ymax - ymin) * (H - 1))

            if 0 <= ix < W and 0 <= iy < H:
                t = i / n_points # normalized iteration count
                color = colorInterpolate(start_color, end_color,
                                         t, mode=render_cfg.interpolation)
                draw.point((ix, iy), fill=color)

        # Save and create dirs, names, etc
        pth = generateFilename(render_cfg.dir, experiment_name, name, render_cfg.interpolation,
                               W, H, render_cfg.iterations)
        # TODO add padding option for non-histogram mode
        img.save(pth, "PNG")
        logging.info(f"saved attractor to ./{pth}")

def compute_attractor_pixel_rate(image_path):
    '''Get proportion of attractor pixels over total image pixels.
    Used to filter chaotic but trivial attractors.
    '''
    img = plt.imread(image_path)
    total_pixels = img.shape[0] * img.shape[1]

    if img.shape[2] == 4:
        # if the image is transparent, count only the non-transparent pixels
        occupied_pixels = np.sum(img[:, :, 3] > 0)
    else:
        # if the image is not transparent, count only the black pixels
        occupied_pixels = np.sum(np.all(img[:, :, :3] == 0, axis=-1))

    return occupied_pixels / total_pixels

# def compute_attractor_pixel_rate(image_path):
#         '''Get proportion of attractor pixels over total image pixels.
#         Used to filter chaotic but trivial attractors.
#         Assumes rgb images (no alpha).
#         '''
#         img = plt.imread(image_path)
#         total_pixels = img.shape[0] * img.shape[1]
#         black_pixels = np.sum(np.all(img[:, :, :3] == 0, axis=-1))
#         return black_pixels / total_pixels

# def compute_attractor_pixel_rate(image_path):
#     '''Get proportion of attractor pixels over total image pixels.
#     Used to filter chaotic but trivial attractors.
#     '''
#     img = plt.imread(image_path)
#     total_pixels = img.shape[0] * img.shape[1]

#     if img.shape[2] == 4:
#         occupied_pixels = np.sum(img[:, :, :3] > 0)
#     else:
#         occupied_pixels = np.sum(np.any(img[:, :, :3] != 0, axis=-1))

#     return occupied_pixels / total_pixels


if __name__ == "__main__":
    ...
