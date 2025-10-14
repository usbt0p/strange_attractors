# Python port of Paul Bourke's lyapunov/gen.c
# By Johan Bichel Lindegaard - http://johan.cc

# found at https://paulbourke.net/fractals/lyapunov/

import argparse
import os
import json
import glob
import logging

import math
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from colors import (colorInterpolate, randomComplementaryColors, 
                    randomTriadColors, randomColoursGoldenRatio)

MAXITERATIONS = 100_000
NEXAMPLES = 10000

def createAttractor(out_path="output"):

    # by default create output directory and attr_number.txt if not exists
    if not os.path.exists(f"{out_path}/attr_number.txt"):
        print(f"Creating output directory {out_path}")
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/attr_number.txt", "x") as f:
            f.write("0")

    with open(f"{out_path}/attr_number.txt", "r") as f:
        file_index = int(f.read().strip())

    for n in tqdm(range(NEXAMPLES), desc="Generating attractors"):
        lyapunov = 0
        xmin = 1e32
        xmax = -1e32
        ymin = 1e32
        ymax = -1e32
        ax, ay, x, y = [], [], [], []

        # Initialize coefficients for this attractor
        for i in range(6):
            ax.append(random.uniform(-2, 2))
            ay.append(random.uniform(-2, 2))

        # Calculate the attractor
        drawit = True
        # This is the initial point, it will serve as the seed for the series
        # Even though the series is chaotic, it is deterministic
        x.append(random.uniform(-0.5, 0.5))
        y.append(random.uniform(-0.5, 0.5))

        # calculate the initial separation between two nearby points
        d0 = -1
        while d0 <= 0: # until the points are not identical
            # explanation: if the points are identical,
            # the distance is zero, and the lyapunov exponent
            # calculation fails (division by zero)
            xe = x[0] + random.uniform(-0.5, 0.5) / 1000.0
            ye = y[0] + random.uniform(-0.5, 0.5) / 1000.0
            dx = x[0] - xe
            dy = y[0] - ye
            d0 = math.sqrt(dx * dx + dy * dy)

        for i in range(1, MAXITERATIONS): # start at 1 to avoid using x[-1]
            # Calculate next term

            x_i = ax[0] + ax[1]*x[i-1] + ax[2]*x[i-1]*x[i-1] + \
                    ax[3]*x[i-1]*y[i-1] + ax[4]*y[i-1] + ax[5]*y[i-1]*y[i-1]
            y_i = ay[0] + ay[1]*x[i-1] + ay[2]*x[i-1]*x[i-1] + \
                    ay[3]*x[i-1]*y[i-1] + ay[4]*y[i-1] + ay[5]*y[i-1]*y[i-1]

            x.append(x_i)
            y.append(y_i)
            xenew = ax[0] + ax[1]*xe + ax[2]*xe*xe + ax[3]*xe*ye + ax[4]*ye + ax[5]*ye*ye
            yenew = ay[0] + ay[1]*xe + ay[2]*xe*xe + ay[3]*xe*ye + ay[4]*ye + ay[5]*ye*ye

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

            # Does the series tend to a point
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                drawit = False
                logging.info("point attractor")
                break

            # Calculate the lyapunov exponents
            # we ignore the first iters to allow the series to settle down into a pattern
            if i > 1000:
                dx = x[i] - xenew
                dy = y[i] - yenew
                dd = math.sqrt(dx * dx + dy * dy)
                lyapunov += math.log(math.fabs(dd / d0))
                xe = x[i] + d0 * dx / dd
                ye = y[i] + d0 * dy / dd

        # TODO check for correctness of lyapunov normalization
        # lyapunov /= (MAXITERATIONS - 1000) # normalize
        # Classify the series according to lyapunov
        if drawit:
            if abs(lyapunov) < 10:
                logging.info("neutrally stable")
                drawit = False
            elif lyapunov < 0:
                print(f"periodic {lyapunov} ")
                drawit = False
            else:
                print(f"chaotic {lyapunov} ")

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
                lyapunov,
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

    with open(f"{out_path}/attr_number.txt", "w") as f:
        f.write(str(file_index))


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
        path, newname := f"{name}{"_" if name else ""}{suffix}.json")):
        suffix += 1
    with open(os.path.join(path, newname + ".json"), "w") as f:
        json.dump(parameters, f, indent=4)

def loadAttractor(pathname):
    with open(f"{pathname}", "r") as f:
        parameters = json.load(f)
    return parameters

def generateAttractorFromParameters(params, iters=MAXITERATIONS
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
        x.append(a[0] + a[1]*x[i-1] + a[2]*x[i-1]*x[i-1] + a[3]*x[i-1]*y[i-1] + a[4]*y[i-1] + a[5]*y[i-1]*y[i-1])
        y.append(b[0] + b[1]*x[i-1] + b[2]*x[i-1]*x[i-1] + b[3]*x[i-1]*y[i-1] + b[4]*y[i-1] + b[5]*y[i-1]*y[i-1])

    return x, y, xmin, xmax, ymin, ymax, iters

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
    maxiterations=MAXITERATIONS,
    interpolation='exp',
    experiment_name="",
    density_sigma=2,  # Gaussian smoothing for histogram mode
    cmap='magma'  # Colormap for histogram mode
):
    """Draw the attractor using the provided parameters.
    Current quirks:
    - if the image already exists, a new name is generated by appending _0, _1, etc
    - if experiment_name is provided, it is used as a subdirectory and prefix for the filename
    - histogram mode ignores start and end colors as well as background color (no transparency), 
        will always use bicubic interpolation, and is limited to matplotlib colormaps
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
            dirpath, newname := f"{name}{"_" if name else ""}{suffix}.{extension}")):
            suffix += 1
        return os.path.join(dirpath, newname)


    if interpolation == 'histogram':
        # Normalize points to [0, 1] range
        xs = (np.array(x) - xmin) / (xmax - xmin)
        ys = (np.array(y) - ymin) / (ymax - ymin)

        # Create a 2D histogram grid
        H2, _, _ = np.histogram2d(xs, ys, bins=[width, height], range=[[0, 1], [0, 1]])
        if np.sum(H2) == 0:
            print(f"Empty histogram for attractor {name}")
            return

        density = gaussian_filter(H2, sigma=density_sigma)
        density[density < 1e-10] = 1e-10 # Avoid zero values before log transform
        img_data = np.log1p(density) # Apply log transform for better dynamic range
        img_data = (img_data - np.min(img_data)) / ( # normalize again to [0, 1]
            np.max(img_data) - np.min(img_data))

        # Create the plot
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(
            img_data.T[::-1],
            origin="lower",
            interpolation="spline16",
            cmap=cmap,  # Use a perceptually uniform colormap
        )
        plt.axis("off")
        plt.tight_layout()

        # Save the image
        pth = generateFilename(dir, experiment_name, name, interpolation, width, height, maxiterations)
        plt.savefig(pth, bbox_inches="tight", pad_inches=0) # TODO aqui
        plt.close()
        print(f"saved attractor to ./{pth}")
    
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
        img.save(pth, "PNG")
        print(f"saved attractor to ./{pth}")


if __name__ == "__main__":

    #createAttractor(out_path="output")

    filenames = glob.glob("output/*.json")
    for filename in tqdm(filenames, desc="Processing files"):
        print(f"Processing {filename}")
        params = loadAttractor(filename)
        x, y, xmin, xmax, ymin, ymax, iters = \
            generateAttractorFromParameters(params, 500_000)
        print("Correctly loaded parameters.")
        name = os.path.basename(filename).split('.')[0]
        drawAttractor(
                name, 
                xmin, xmax,
                ymin, ymax,
                x, y,
                maxiterations=500_000,
                width=1500,
                height=1500,
                start_color=(0, 0, 0),
                end_color=(255, 255, 255),
                background_color=(125, 125, 125),
                dir="para_github",
                interpolation="histogram",
            )   