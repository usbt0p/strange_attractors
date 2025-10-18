import argparse
import os
import json
import glob
import logging
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import math
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from colors import (colorInterpolate, randomComplementaryColors, 
                    randomTriadColors, randomColoursGoldenRatio)

def createAttractor(examples, iterations, out_path="output"):
    '''Generate strange attractors using random coefficients and save them as images and JSON files.
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

            # random coefficients
            ax = [random.uniform(-2, 2) for _ in range(6)]
            ay = [random.uniform(-2, 2) for _ in range(6)]
            # This is the initial point, it will serve as the seed for the series
            # Even though the series is chaotic, it is deterministic
            x = [random.uniform(-0.5, 0.5)]
            y = [random.uniform(-0.5, 0.5)]
            # min and max bounds
            xmin = ymin = 1e32
            xmax = ymax = -1e32
            lyapunov = 0.0     
            
            # calculate the initial separation between two nearby points and ensure it's not zero
            d0 = -1
            while d0 <= 0: # until the points are not identical
                # xe and ye are coords of our initial point, but slightly perturbed
                xe = x[0] + random.uniform(-0.5, 0.5) / 1000.0
                ye = y[0] + random.uniform(-0.5, 0.5) / 1000.0
                # dx and dy aid us in later filtering point or infinite attractors,
                # they are the vector between the seed and the perturbed point
                dx = x[0] - xe
                dy = y[0] - ye
                # this is just the distance of the previous vector, 
                # a theoretical infinitesimal that will be used to renormalize the
                # distance between the two points after each iteration
                d0 = math.sqrt(dx * dx + dy * dy) # (epsilon in Bennetin method)

            drawit = True
            # Calculate the attractor
            for i in range(1, iterations): # start at 1 to avoid using x[-1]
                # Calculate next term

                x_i = ax[0] + ax[1]*x[i-1] + ax[2]*x[i-1]**2 + \
                        ax[3]*x[i-1]*y[i-1] + ax[4]*y[i-1] + ax[5]*y[i-1]**2
                y_i = ay[0] + ay[1]*x[i-1] + ay[2]*x[i-1]**2 + \
                        ay[3]*x[i-1]*y[i-1] + ay[4]*y[i-1] + ay[5]*y[i-1]**2

                x.append(x_i)
                y.append(y_i)
                # this represents a nearby point, but slightly separated, on another orbit
                # we'll use this to calculate the lyapunov exponent
                xenew = ax[0] + ax[1]*xe + ax[2]*xe**2 + ax[3]*xe*ye + ax[4]*ye + ax[5]*ye**2
                yenew = ay[0] + ay[1]*xe + ay[2]*xe**2 + ay[3]*xe*ye + ay[4]*ye + ay[5]*ye**2

                # Update the bounds
                xmin = min(xmin, x[i])
                ymin = min(ymin, y[i])
                xmax = max(xmax, x[i])
                ymax = max(ymax, y[i])

                # Does the series tend to infinity
                if xmin < -1e10 or ymin < -1e10 or xmax > 1e10 or ymax > 1e10:
                    drawit = False
                    logging.info("infinite attractor")
                    break

                # if the vector between current and prev point is too small, 
                # the series will tend to a point
                dx = x[i] - x[i - 1]
                dy = y[i] - y[i - 1]
                if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                    drawit = False
                    logging.info("point attractor")
                    break

                # Calculate the lyapunov exponents using 
                # we ignore the first iters to allow the series to settle down into a pattern
                if i > 1000:
                    # this approach of approximating the lyapunov exponent is called the Bennetin method
                    # it's not 100% equal to the mathematical definition, but it's a good approximation

                    # two vectors are computed as the difference between the current point and
                    # our perturbed point (xenew, yenew)
                    dx = x[i] - xenew
                    dy = y[i] - yenew
                    dd = math.sqrt(dx * dx + dy * dy) # compute the distance between the norm
                    # add the log of the ratio between the two current point's distances and the initial infinitesimal distance
                    lyapunov += math.log(math.fabs(dd / d0))
                    # renormalize the perturbed point to be d0 away from the current point (to avoid overflow or underflow of the value)
                    xe = x[i] + d0 * dx / dd
                    ye = y[i] + d0 * dy / dd

                # TODO check for correctness of lyapunov normalization
                # lyapunov /= (MAXITERATIONS - 1000) # normalize
                # and what about T? 

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

def generateAttractorFromParameters(params, iters
                                ) -> tuple[list, list, float, float, float, float, int]:
    equation = params.get("equation", None)
    a = params["coefficients"]["a"]
    b = params["coefficients"]["b"]
    xmin = params["bounds"]["xmin"]
    xmax = params["bounds"]["xmax"]
    ymin = params["bounds"]["ymin"]
    ymax = params["bounds"]["ymax"]
    x0 = params["seed_point"]["x"]
    y0 = params["seed_point"]["y"]

    x, y = [], []
    x.append(x0)
    y.append(y0)

    for i in range(1, iters):
        x.append(a[0] + a[1]*x[i-1] + a[2]*x[i-1]**2 + a[3]*x[i-1]*y[i-1] + a[4]*y[i-1] + a[5]*y[i-1]**2)
        y.append(b[0] + b[1]*x[i-1] + b[2]*x[i-1]**2 + b[3]*x[i-1]*y[i-1] + b[4]*y[i-1] + b[5]*y[i-1]**2)

    return x, y, xmin, xmax, ymin, ymax

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
    interpolation='exp',
    experiment_name="",
    density_sigma=0,  # Gaussian smoothing for histogram mode
    cmap='magma',  # Colormap for histogram mode
    pad_size=0,
):
    """Draw the attractor using the provided parameters.
    Current quirks:
    - if the image already exists, a new name is generated by appending _0, _1, etc
    - if experiment_name is provided, it is used as a subdirectory and prefix for the filename
    - histogram mode ignores start and end colors, will always use bicubic interpolation
    - padding only works in histogram mode
    """            

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


    if interpolation == 'histogram':
        # Normalize points to [0, 1] range
        xs = (np.array(x) - xmin) / (xmax - xmin)
        ys = (np.array(y) - ymin) / (ymax - ymin)

        # Create a 2D histogram grid
        H2, _, _ = np.histogram2d(xs, ys, bins=[width, height], range=[[0, 1], [0, 1]])
        if np.sum(H2) == 0:
            logging.debug(f"Empty histogram for attractor {name}")
            return

        density = H2
        if density_sigma > 0: density = gaussian_filter(H2, sigma=density_sigma)
        density[density < 1e-10] = 1e-10 # Avoid zero values before log transform
        img_data = np.log1p(density) # Apply log transform for better dynamic range
        img_data = (img_data - np.min(img_data)) / ( # normalize again to [0, 1]
            np.max(img_data) - np.min(img_data))

        # Create the plot
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(
            # "rotate" the histogram to match coords, although the orientation is not really important
            img_data.T[::-1], 
            origin="lower",
            interpolation="bicubic",
            cmap=cmap,  # Use a perceptually uniform colormap
        )
        plt.axis("off")
        plt.tight_layout()

        # Save the image
        pth = generateFilename(dir, experiment_name, name, interpolation, width, height, maxiterations)
        plt.savefig(pth, bbox_inches="tight", pad_inches=pad_size, facecolor='black')
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
    RENDER_ITERATIONS = 20_000_000 # long processing but good rendering
    base_dir = "set_num"

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
    #createAttractor(out_path=f"{base_dir}/out", examples=EXAMPLES, iterations=CREATE_ITERATIONS)

    cmaps = ['plasma', 'magma', 'viridis', 'cividis', 
             'turbo', 'summer', 'autumn', 'winter', 'copper']
    recolor = {'n':1, 'color':(255,255,255), 'resample_size':512}
    
    for cmap_name in cmaps:
        print(f"Rendering with colormap {cmap_name}...")
        render_kwargs['cmap'] = create_linear_colormap(preset=cmap_name, 
                                                       recolor_base=recolor)

        # rendering multiple attracctor is a perfect task for multiprocessing! cpu go brr
        results = draw_attractors_in_parallel(
            glob.glob(f"{base_dir}/out/*.json"), 
            RENDER_ITERATIONS, 
            n_processes=cpu_count() - 4, # leave some CPU for other tasks
            batch_size=50,
            **render_kwargs
        )

    render_kwargs['cmap'] = create_linear_colormap(colors=[(255, 0, 0),(0, 255, 0)] ,
                                                    recolor_base=recolor)
    results = draw_attractors_in_parallel(
            glob.glob(f"{base_dir}/out/*.json"), 
            RENDER_ITERATIONS, 
            n_processes=cpu_count() - 4, # leave some CPU for other tasks
            batch_size=50,
            **render_kwargs
        )

