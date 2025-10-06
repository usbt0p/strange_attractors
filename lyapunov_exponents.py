# Python port of Paul Bourke's lyapunov/gen.c
# By Johan Bichel Lindegaard - http://johan.cc

# found at https://paulbourke.net/fractals/lyapunov/

import math
import random
from PIL import Image, ImageDraw, ImageFilter
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description="Search for chaos.")
# parser.add_argument('-i', dest='maxiterations' metavar='N', type=int,
#            help='Maximum iterations.')

args = parser.parse_args()

MAXITERATIONS = 500_000
NEXAMPLES = 1000


def createAttractor():
    for n in range(NEXAMPLES):
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
        x.append(random.uniform(-0.5, 0.5))
        y.append(random.uniform(-0.5, 0.5))

        d0 = -1
        while d0 <= 0:
            xe = x[0] + random.uniform(-0.5, 0.5) / 1000.0
            ye = y[0] + random.uniform(-0.5, 0.5) / 1000.0
            dx = x[0] - xe
            dy = y[0] - ye
            d0 = math.sqrt(dx * dx + dy * dy)

        for i in range(MAXITERATIONS):
            # Calculate next term

            #
            x.append(
                ax[0]
                + ax[1] * x[i - 1]
                + ax[2] * x[i - 1] * x[i - 1]
                + ax[3] * x[i - 1] * y[i - 1]
                + ax[4] * y[i - 1]
                + ax[5] * y[i - 1] * y[i - 1]
            )
            y.append(
                ay[0]
                + ay[1] * x[i - 1]
                + ay[2] * x[i - 1] * x[i - 1]
                + ay[3] * x[i - 1] * y[i - 1]
                + ay[4] * y[i - 1]
                + ay[5] * y[i - 1] * y[i - 1]
            )
            xenew = (
                ax[0]
                + ax[1] * xe
                + ax[2] * xe * xe
                + ax[3] * xe * ye
                + ax[4] * ye
                + ax[5] * ye * ye
            )
            yenew = (
                ay[0]
                + ay[1] * xe
                + ay[2] * xe * xe
                + ay[3] * xe * ye
                + ay[4] * ye
                + ay[5] * ye * ye
            )

            # Update the bounds
            xmin = min(xmin, x[i])
            ymin = min(ymin, y[i])
            xmax = max(xmax, x[i])
            ymax = max(ymax, y[i])

            # Does the series tend to infinity
            if xmin < -1e10 or ymin < -1e10 or xmax > 1e10 or ymax > 1e10:
                drawit = False
                print("infinite attractor")
                break

            # Does the series tend to a point
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                drawit = False
                print("point attractor")
                break

            # Calculate the lyapunov exponents
            if i > 1000:
                dx = x[i] - xenew
                dy = y[i] - yenew
                dd = math.sqrt(dx * dx + dy * dy)
                lyapunov += math.log(math.fabs(dd / d0))
                xe = x[i] + d0 * dx / dd
                ye = y[i] + d0 * dy / dd

        # Classify the series according to lyapunov
        if drawit:
            if abs(lyapunov) < 10:
                print("neutrally stable")
                drawit = False
            elif lyapunov < 0:
                print("periodic {} ".format(lyapunov))
                drawit = False
            else:
                print("chaotic {} ".format(lyapunov))

        # Save the image
        if drawit:
            saveAttractor(
                str(n), "output", ax, ay, xmin, xmax, ymin, ymax
            )
            drawAttractor(
                n, xmin, xmax, ymin, ymax, x, y, dir="output", 
                maxiterations=MAXITERATIONS,
            )


def saveAttractor(
    name,
    dir,
    a,
    b,
    xmin,
    xmax,
    ymin,
    ymax,
):
    
    path = f"{dir}/{name}.txt"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # check if file exists and if so find a new name
    if os.path.exists(path):
        suffix = 1
        while os.path.exists(f"{dir}/{name}_{suffix}.txt"):
            suffix += 1
        path = f"{dir}/{name}_{suffix}.txt"

    # Save the parameters
    with open(path, "w") as f:
        f.write("{} {} {} {}\n".format(xmin, ymin, xmax, ymax))
        for i in range(6):
            f.write("{} {}\n".format(a[i], b[i]))


def colorInterpolate(color1, color2, t, mode='linear'):
    match mode:
        case 'lin': pass  # t is already linear
        case 'log': t = math.log(t * (math.e - 1) + 1)  # log scale
        case 'exp': t = (math.exp(t) - 1) / (math.e - 1)  # exponential scale
        case 'cos': t = (1 - math.cos(math.pi * t)) / 2  # cosine scale
        case 'bicubic': t = t * t * (3 - 2 * t)  # smoothstep
        case _: pass  # default to linear if unknown mode

    r = int(color1[0] + t * (color2[0] - color1[0]))
    g = int(color1[1] + t * (color2[1] - color1[1]))
    b = int(color1[2] + t * (color2[2] - color1[2]))
    
    return (r, g, b)

def drawAttractor(
    n,
    xmin,
    xmax,
    ymin,
    ymax,
    x,
    y,
    width=800,
    height=800,
    start_color=(255, 0, 0),
    end_color=(255, 255, 255),
    background_color=None,
    dir="output",
    maxiterations=MAXITERATIONS,
    interpolation='linear',
    experiment_name="",
    density_sigma=2,  # Gaussian smoothing for histogram mode
    cmap='magma'  # Colormap for histogram mode
):
    # Validate x and y ranges
    if xmax <= xmin or ymax <= ymin:
        print(f"Invalid ranges for attractor {n}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
        return

    # Validate x and y values
    for i in range(len(x)):
        if math.isnan(x[i]) or math.isnan(y[i]):
            #print(f"NaN detected in x or y values at index {i} for attractor {n}")
            # imput value with previous value or 0 if first
            if i == 0:
                x[i] = 0
                y[i] = 0
            else:
                x[i] = x[i-1]
                y[i] = y[i-1]
            

    if interpolation == 'histogram':
        # TODO add transparency option for histogram mode
        # Ensure valid ranges
        if xmin == xmax or ymin == ymax:
            print(f"Invalid range for attractor {n}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
            return

        # Normalize points to [0, 1] range
        xs = (np.array(x) - xmin) / (xmax - xmin)
        ys = (np.array(y) - ymin) / (ymax - ymin)

        # Create a 2D histogram grid
        H2, _, _ = np.histogram2d(xs, ys, bins=[width, height], range=[[0, 1], [0, 1]])

        # Check if the histogram is empty
        if np.sum(H2) == 0:
            print(f"Empty histogram for attractor {n}")
            return

        # Apply Gaussian smoothing
        density = gaussian_filter(H2, sigma=density_sigma)
        # Avoid zero values before log transform
        density[density < 1e-10] = 1e-10
        # Apply log transform for better dynamic range
        img_data = np.log1p(density)
        # Normalize the image data to [0, 1] for better visualization
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Create the plot
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(
            img_data.T[::-1],
            origin="lower",
            interpolation="bicubic",
            cmap=cmap,  # Use a perceptually uniform colormap
        )
        plt.axis("off")
        plt.tight_layout()

        # Save the image
        if experiment_name:
            n = f"{n}_histogram_{width}x{height}_{maxiterations}"

        if not os.path.exists(f"./{dir}/{experiment_name}"):
            os.makedirs(f"./{dir}/{experiment_name}")
        plt.savefig(f"./{dir}/{experiment_name}/{n}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"saved attractor to ./{dir}/{experiment_name}/{n}.png")
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
                t = i / maxiterations
                color = colorInterpolate(start_color, end_color, t, mode=interpolation)
                draw.point((ix, iy), fill=color)

        # Save the image
        if experiment_name:
            n = f"{n}_{interpolation}_{maxiterations}_{width}x{height}_{maxiterations}_{start_color}_{end_color}"

        if not os.path.exists(f"./{dir}/{experiment_name}"):
            os.makedirs(f"./{dir}/{experiment_name}")

        # check if file exists and if so find a new name
        if os.path.exists(f"./{dir}/{experiment_name}/{n}.png"):
            suffix = 1
            while os.path.exists(f"./{dir}/{experiment_name}/{n}_{suffix}.png"):
                suffix += 1
            n = f"{n}_{suffix}"   

        img.save(f"./{dir}/{experiment_name}/{n}.png", "PNG")
        
        print(f"saved attractor to ./{dir}/{experiment_name}/{n}.png")


def loadAttractor(n, maxiterations=MAXITERATIONS):

    # Load the parameters
    with open("output/{}.txt".format(n), "r") as f:
        line = f.readline()
        xmin, ymin, xmax, ymax = map(float, line.split())
        a, b = [], []
        for i in range(6):
            line = f.readline()
            ai, bi = map(float, line.split())
            a.append(ai)
            b.append(bi)
        f.close()

    # Regenerate x and y values
    x, y = [], []
    x.append(random.uniform(-0.5, 0.5))
    y.append(random.uniform(-0.5, 0.5))
    for i in range(1, maxiterations):
        
        x_i = (
            a[0]
            + a[1] * x[i - 1]
            + a[2] * x[i - 1] * x[i - 1]
            + a[3] * x[i - 1] * y[i - 1]
            + a[4] * y[i - 1]
            + a[5] * y[i - 1] * y[i - 1]
        )

        x.append(x_i)
        
        
        y_i = (b[0]
            + b[1] * x[i - 1]
            + b[2] * x[i - 1] * x[i - 1]
            + b[3] * x[i - 1] * y[i - 1]
            + b[4] * y[i - 1]
            + b[5] * y[i - 1] * y[i - 1])
        
        y.append(y_i)

    # FIXME the problem is here most likely, since the generation works
    # since chaotic attractors are sensitive to initial conditions
    # regenerating the x and y values might lead to divergence, which we see since
    # most of the problematic attractors collapse into a point attractor
    # a possible solution is to store the initial x and y values in the file, but that
    # is not how the original code worked, and leads to large files, + you cant add iterations

    #     # check for nan
    #     n_nans = 0
    #     if math.isnan(x_i) or math.isnan(y_i):
    #         #print(f"NaN detected in x at iteration {i} for attractor {n}, imputing with previous value")
    #         if i == 0:
    #             x[i] = 0
    #             y[i] = 0
    #             print("first value is nan, setting to 0")
    #         else:
    #             x[i] = x[i-1]
    #             y[i] = y[i-1]
    #             n_nans += 1
    # print(f"Total NaNs encountered during regeneration: {n_nans}")
            

    return a, b, xmin, xmax, ymin, ymax, x, y

def randomComplementaryColors(lbound=0, ubound=255):
    # Generate a random color within bounds
    r = random.randint(lbound, ubound)
    g = random.randint(lbound, ubound)
    b = random.randint(lbound, ubound)

    # Calculate its complementary color within bounds
    comp_r = ubound + lbound - r
    comp_g = ubound + lbound - g
    comp_b = ubound + lbound - b

    # Clamp to bounds
    comp_r = max(lbound, min(ubound, comp_r))
    comp_g = max(lbound, min(ubound, comp_g))
    comp_b = max(lbound, min(ubound, comp_b))

    return (r, g, b), (comp_r, comp_g, comp_b)
    
def randomTriadColors(lbound=0, ubound=255):
    # Generate a random base color within bounds
    r = random.randint(lbound, ubound)
    g = random.randint(lbound, ubound)
    b = random.randint(lbound, ubound)

    # Calculate two triadic colors within bounds
    span = ubound - lbound + 1
    triad1_r = lbound + ((r - lbound + span // 3) % span)
    triad1_g = lbound + ((g - lbound + span // 3) % span)
    triad1_b = lbound + ((b - lbound + span // 3) % span)

    triad2_r = lbound + ((r - lbound + 2 * span // 3) % span)
    triad2_g = lbound + ((g - lbound + 2 * span // 3) % span)
    triad2_b = lbound + ((b - lbound + 2 * span // 3) % span)

    # coin toss for the second color
    if random.random() < 0.5:
        triad = (triad1_r, triad1_g, triad1_b)
    else:
        triad = (triad2_r, triad2_g, triad2_b)

    return (r, g, b), triad

def randomColoursGoldenRatio(lbound=0, ubound=255):
    # Generate a random base color within bounds
    r = random.randint(lbound, ubound)
    g = random.randint(lbound, ubound)
    b = random.randint(lbound, ubound)

    # Calculate a second color using the golden ratio
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    span = ubound - lbound + 1

    golden_r = lbound + int(((r - lbound) * phi) % span)
    golden_g = lbound + int(((g - lbound) * phi) % span)
    golden_b = lbound + int(((b - lbound) * phi) % span)

    return (r, g, b), (golden_r, golden_g, golden_b)

if __name__ == "__main__":

    #createAttractor()

    #ex = [54, 93, 99, 107, 139, "210_1", 542, 620, 6, 210, "989_1", 511, 836, 107, 224, 292, 756, 903, 814, 298, 383]
    problem_ex = [6, 107, 210, 511, 620]
    
    #backgrounds = [None, (0,0,0,255), (255,255,255,255)]      
    sizes = [(400,400)]#[(400, 400), (800, 800), (1000, 1000), (1500, 1500)]
    iter = 2_000_000
    cmaps = ["magma", "inferno", "plasma", "viridis", "cividis"]

    for num in problem_ex:
        
        # eat a and b cause we dont need them
        _, _, *attractor_params, = loadAttractor(num, maxiterations=iter)

        # log info about the attractor
        print(f"Attractor {num}: params={len(attractor_params)}")
        print(f"len of x and y: {len(attractor_params[-2])}, {len(attractor_params[-1])}")
        # see mean and other stats of params
        xmin, xmax, ymin, ymax, x, y = attractor_params
        # print stats and analytics about params
        print(f"x: {x[:5]}, y: {y[:5]}")
        print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")
        print()
        
        c1, c2 = randomComplementaryColors()

        for size in sizes:
            
            
            drawAttractor(
                num, #903,
                *attractor_params,
                dir="renders",
                width=size[0],
                height=size[1],
                maxiterations=iter,
                end_color=c1,
                start_color=c2,
                #background_color=background,
                interpolation='histogram',
                experiment_name=f"problematic",
                cmap='magma'
            )

    # TODO
    #https://www.nathanselikoff.com/training/tutorial-strange-attractors-in-c-and-opengl
    #https://www3.fi.mdp.edu.ar/fc3/SisDin2009/Clase%202/nousado/CHAOSCOPE/help/en/manual/attractors.htm
    #http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/