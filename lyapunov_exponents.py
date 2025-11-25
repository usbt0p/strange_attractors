import os
import json
import glob
import logging

import numba
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from colors import (colorInterpolate, randomComplementaryColors, 
                    randomTriadColors, randomColoursGoldenRatio)

import numpy as np
import numba
import math
import random

@numba.njit(fastmath=True)
def compute_lyapunov(iterations, discard=1000):
    """Compute the Lyapunov exponent of the attractor defined by coefficients a and b.
    a and b are arrays of coefficients, x0 and y0 are the seed points.
    discard: number of initial iterations to discard for transient behavior.
    """
    # random coefficients
    ax = np.array([random.uniform(-2.0, 2.0) for _ in range(6)])
    ay = np.array([random.uniform(-2.0, 2.0) for _ in range(6)])
    
    # cache coefficients in local variables for speed (cpu registers)
    ax0, ax1, ax2, ax3, ax4, ax5 = ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]
    ay0, ay1, ay2, ay3, ay4, ay5 = ay[0], ay[1], ay[2], ay[3], ay[4], ay[5]

    # This is the initial point, it will serve as the seed for the series
    # Even though the series is chaotic, it is deterministic
    x_val = random.uniform(-0.5, 0.5)
    y_val = random.uniform(-0.5, 0.5)
    
    # Usamos listas, que son eficientes para "Early Exit" (si el atractor falla rápido)
    x = [x_val]
    y = [y_val]

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
        xe = x_val + random.uniform(-0.5, 0.5) / 1000.0
        ye = y_val + random.uniform(-0.5, 0.5) / 1000.0
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
        
        # Calculate next term
        # direct multiplication is theoretically faster than pow
        prev_x_sq = prev_x * prev_x
        prev_y_sq = prev_y * prev_y
        xy_term = prev_x * prev_y

        x_i = ax0 + ax1*prev_x + ax2*prev_x_sq + ax3*xy_term + ax4*prev_y + ax5*prev_y_sq
        y_i = ay0 + ay1*prev_x + ay2*prev_x_sq + ay3*xy_term + ay4*prev_y + ay5*prev_y_sq

        x.append(x_i)
        y.append(y_i)
        
        # this represents a nearby point, but slightly separated, on another orbit
        # we'll use this to calculate the lyapunov exponent
        xe_sq = xe * xe
        ye_sq = ye * ye
        xeye = xe * ye
        xenew = ax0 + ax1*xe + ax2*xe_sq + ax3*xeye + ax4*ye + ax5*ye_sq
        yenew = ay0 + ay1*xe + ay2*xe_sq + ay3*xeye + ay4*ye + ay5*ye_sq

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

    return lyapunov, drawit, x, y, ax, ay, xmin, xmax, ymin, ymax

def createAttractors(examples, iterations, out_path="output"):
    '''Create strange attractors and save them as images and JSON files.
    If you want more insights into the algorithm: 
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7512692/#sec3-entropy-20-00175
    https://paulbourke.net/fractals/lyapunov/
    '''

    # by default create output directory and attr_number.txt if not exists
    if not os.path.exists(f"{out_path}/attr_number.txt"):
        print(f"Creating output directory {out_path}")
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/attr_number.txt", "x") as f:
            f.write("0")

    with open(f"{out_path}/attr_number.txt", "r") as f:
        file_index = int(f.read().strip())

    pbar = tqdm(total=examples, desc="Generating attractors")
    try: # catch keyboard interrupt to save progress
        while file_index < examples:

            lyapunov, drawit, x, y, ax, ay, xmin, xmax, ymin, ymax = compute_lyapunov(
                iterations=iterations, discard=1000,
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
                    None,
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
                    width=500,
                    height=500,
                    dir=out_path,
                    interpolation=None,
                )
                file_index += 1
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
        "equation": equation,
        "bounds": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        },
        "coefficients": {
            "a": [a[i] for i in range(6)],
            "b": [b[i] for i in range(6)]
        },
        "seed_point": {
            "x": x_o,
            "y": y_o
        },
        "lyapunov": lyapunov
    }

    suffix = 0
    while os.path.exists(os.path.join(
        path, newname := f"{name}{"_" if str(name) else ""}{suffix}.json")):
        suffix += 1
    with open(os.path.join(path, newname), "w") as f:
        json.dump(parameters, f, indent=4)

def loadAttractor(pathname):
    with open(f"{pathname}", "r") as f:
        parameters = json.load(f)
    return parameters

@numba.njit(fastmath=True)
def compute_attractor_core(a, b, x0, y0, iters):
    """Core function to compute the attractor points using Numba for speed.
    a and b are arrays of coefficients, x0 and y0 are the seed points."""
    # preallocate memory
    x = np.zeros(iters)
    y = np.zeros(iters)
    x[0] = x0
    y[0] = y0

    for i in range(1, iters):
        prev_x = x[i-1]
        prev_y = y[i-1]
        x[i] = a[0] + a[1]*prev_x + a[2]*prev_x**2 + a[3]*prev_x*prev_y + a[4]*prev_y + a[5]*prev_y**2
        y[i] = b[0] + b[1]*prev_x + b[2]*prev_x**2 + b[3]*prev_x*prev_y + b[4]*prev_y + b[5]*prev_y**2
    return x, y

def generateAttractorFromParameters(params, iters):
    '''Interface to generate attractor points from loaded parameters.'''

    # ensure numpy for numba
    a = np.array(params["coefficients"]["a"], dtype=np.float64)
    b = np.array(params["coefficients"]["b"], dtype=np.float64)
    
    xmin = params["bounds"]["xmin"]
    xmax = params["bounds"]["xmax"]
    ymin = params["bounds"]["ymin"]
    ymax = params["bounds"]["ymax"]
    x0 = params["seed_point"]["x"]
    y0 = params["seed_point"]["y"]

    x, y = compute_attractor_core(a, b, x0, y0, iters)

    logging.info("Attractor generated from parameters.")
    return x, y, xmin, xmax, ymin, ymax

def computeDensity(
    xmin, xmax,
    ymin, ymax,
    x, y,
    width, height,
    density_sigma,  # Gaussian smoothing for histogram mode
):
    """Compute the log-density histogram of the attractor points.
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
    # "rotate" the histogram to match coords, although the orientation is not really important
    return img_data.T[::-1]

def drawAttractor(
    name,
    xmin, xmax,
    ymin, ymax,
    x, y,
    width=800,
    height=800,
    start_color=(0, 0, 0),
    end_color=(0, 0, 0),
    background_color=None,
    dir="output",
    maxiterations=100_000,
    interpolation='histogram',
    experiment_name="",
    density_sigma=0,  # Gaussian smoothing for histogram mode
    cmap='magma',  # Colormap for histogram mode
    pad_size=0,
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

    def generateFilename(dir, experiment_name, name, interpolation, 
                         width, height, maxiterations, extension="png"):
        if not os.path.exists(dirpath := os.path.join(dir, experiment_name)):
            os.makedirs(dirpath, exist_ok=True) # ignoring experiment_name if empty is intentional
        if experiment_name:
            name = f"{name}_{interpolation}_{width}x{height}_{maxiterations}"
        # check if file exists and if so find a new name
        suffix = 0
        while os.path.exists(os.path.join(
            dirpath, newname := f"{name}{"_" if str(name) else ""}{suffix}.{extension}")):
            suffix += 1
        return os.path.join(dirpath, newname)


    if interpolation == 'histogram':
        
        img_data = computeDensity(
            xmin, xmax,
            ymin, ymax,
            x, y,
            width, height,
            density_sigma,
        )

        # Create the plot
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(
            img_data, 
            origin="lower",
            interpolation="bicubic",
            cmap=cmap,
        )
        plt.axis("off")
        plt.tight_layout()

        # Save the image
        pth = generateFilename(dir, experiment_name, name, interpolation, width, height, maxiterations)
        # TODO for alpha, something like np.array((*color, 1.0))??
        background = tuple(c / 255 for c in background_color) if background_color else tuple(cmap(0)[:3])
        plt.savefig(pth, bbox_inches="tight", pad_inches=pad_size, facecolor=background)
        plt.close()
        logging.info(f"saved attractor to ./{pth}")
    
    else:
        if background_color:
            img = Image.new("RGBA", (width, height), background_color)
        else:
            img = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(img)

        # iterate through all points and draw them with the color interpolated
        for i in range(1, maxiterations):
            # Map the point to image coordinates
            ix = int((x[i] - xmin) / (xmax - xmin) * (width - 1))
            iy = int((y[i] - ymin) / (ymax - ymin) * (height - 1))
            
            if 0 <= ix < width and 0 <= iy < height:
                t = i / maxiterations # normalized iteration count
                color = colorInterpolate(start_color, end_color, t, mode=interpolation)
                draw.point((ix, iy), fill=color)

        # Save and create dirs, names, etc
        pth = generateFilename(dir, experiment_name, name, interpolation, width, height, maxiterations)
        # TODO add padding option for non-histogram mode
        img.save(pth, "PNG")
        logging.info(f"saved attractor to ./{pth}")


if __name__ == "__main__":
    from parallel import draw_attractors_in_parallel, draw_single_attractor
    from colors import create_linear_colormap

    CREATE_ITERATIONS = 100_000
    EXAMPLES = 100
    RENDER_ITERATIONS = 25_000_000 # long processing but good rendering

    base_dir = "numba"

    attractors = [4750, 4755, 4761, 4827, 4866, 4894, 4909, 4917, 4944, 
                  4979, 5009, 5014, 5043, 5059, 5061, 5072, 5094, 5095, 
                  5154, 5171, 5188, 5279, 5352, 5354, 5363, 5385, 5454, 5455]

    input_files = [f"new_attractors/out/{a}_0.json.json" for a in attractors]

    render_kwargs = {
        "width": 1000,
        "height": 1000,
        "cmap": create_linear_colormap(preset='viridis'),
        "dir": f"{base_dir}/renders",
        "interpolation": "histogram",
        "density_sigma": 0,
        "pad_size": 0.5
    }
    #logging.basicConfig(level=logging.INFO)
    
    # TODO parallelize this too
    # find EXAMPLES random attractors with CREATE_ITERATIONS iters for each one
    createAttractors(out_path=f"{base_dir}/out", examples=EXAMPLES, iterations=CREATE_ITERATIONS)

    recolor = {'n':1, 'color':(0,0,0), 'resample_size':512}

    # give more space to the last colour
    # TODO fix this
    positions = [0.0, 0.35, 1.0]

    # customs = [
    #     [(0,0,0), (2, 24, 107), (255, 145, 0)],
    #     [(255, 255, 255), (190, 255, 147), (0, 86, 161)],
    #     [(255, 255, 255), (192, 255, 57), (255, 92, 122)],
    # ]

    # for custom in customs:
    #     print(f"Rendering with custom colormap {custom}...")
    #     recolor['color'] = custom[0]

    #     render_kwargs['cmap'] = create_linear_colormap(colors=custom,
    #                                                    positions=positions,
    #                                                     recolor_base=recolor)
    #     results = draw_attractors_in_parallel(
    #             input_files, 
    #             RENDER_ITERATIONS, 
    #             n_processes=cpu_count() - 6, # leave some CPU for other tasks
    #             batch_size=10,
    #             **render_kwargs
    #         )
    
    print("CUSTOMS DONE")
    input("Press Enter to continue to preset colormaps...")

    cmaps = ['magma', 'viridis', 'cividis', 
             'turbo', 'summer', 'autumn', 
             'winter', 'copper', 'afmhot', 
             'bone', 'PuOr', 'berlin', 'managua']
    recolor['color'] = (0,0,0)
        
    for cmap_name in cmaps:
        print(f"Rendering with colormap {cmap_name}...")
        render_kwargs['cmap'] = create_linear_colormap(preset=cmap_name, 
                                                       recolor_base=recolor)

        # rendering multiple attracctor is a perfect task for multiprocessing! cpu go brr
        results = draw_attractors_in_parallel(
            input_files, 
            RENDER_ITERATIONS, 
            n_processes=os.cpu_count() - 6, # leave some CPU for other tasks
            batch_size=10,
            **render_kwargs
        )



