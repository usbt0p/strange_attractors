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
    density_sigma=2,  # New parameter for Gaussian smoothing
):
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
        x.append(
            a[0]
            + a[1] * x[i - 1]
            + a[2] * x[i - 1] * x[i - 1]
            + a[3] * x[i - 1] * y[i - 1]
            + a[4] * y[i - 1]
            + a[5] * y[i - 1] * y[i - 1]
        )
        y.append(
            b[0]
            + b[1] * x[i - 1]
            + b[2] * x[i - 1] * x[i - 1]
            + b[3] * x[i - 1] * y[i - 1]
            + b[4] * y[i - 1]
            + b[5] * y[i - 1] * y[i - 1]
        )

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

    ex = [542, 620, 6, 210, "989_1", 511, 836] #[107, 224, 292, 756, 903, 814]
    #backgrounds = [None, (0,0,0,255), (255,255,255,255)]      
    sizes = [(1500,1500)]#[(400, 400), (800, 800), (1500, 1500), (2000, 2000)]
    iter = 10_000_000

    for num in ex:
        
        # eat a and b cause we dont need them
        _, _, *attractor_params, = loadAttractor(num, maxiterations=iter)
        
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
                interpolation='exp',
                experiment_name=f"test/{size}"
            )

    # TODO
    #https://www.nathanselikoff.com/training/tutorial-strange-attractors-in-c-and-opengl
    #https://www3.fi.mdp.edu.ar/fc3/SisDin2009/Clase%202/nousado/CHAOSCOPE/help/en/manual/attractors.htm
    #http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/