# Python port of Paul Bourke's lyapunov/gen.c
# By Johan Bichel Lindegaard - http://johan.cc

# from https://paulbourke.net/fractals/lyapunov/

import math
import random
from PIL import Image, ImageDraw
import argparse
import json
import os
import glob

parser = argparse.ArgumentParser(description='Search for chaos.')
# parser.add_argument('-i', dest='maxiterations' metavar='N', type=int,
#            help='Maximum iterations.')

args = parser.parse_args()

MAXITERATIONS = 100000
NEXAMPLES = 10000

def createAttractor(out_path="output"):
    
    if not os.path.exists(f"{out_path}/attr_number.txt"):
        print(f"Creating output directory {out_path}")
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/attr_number.txt", "x") as f:
            f.write("0")

    # read the file that holds the number of the last attractor
    with open(f"{out_path}/attr_number.txt", "r") as f:
        file_index = int(f.read().strip())

    for n in range(NEXAMPLES):      
        lyapunov = 0
        xmin= 1e32
        xmax=-1e32
        ymin= 1e32
        ymax=-1e32
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

        for i in range(1, MAXITERATIONS):
            # Calculate next term
            x_i = ax[0] + ax[1]*x[i-1] + ax[2]*x[i-1]*x[i-1] + ax[3]*x[i-1]*y[i-1] + ax[4]*y[i-1] + ax[5]*y[i-1]*y[i-1]
            y_i = ay[0] + ay[1]*x[i-1] + ay[2]*x[i-1]*x[i-1] + ay[3]*x[i-1]*y[i-1] + ay[4]*y[i-1] + ay[5]*y[i-1]*y[i-1]
            x.append(x_i)
            y.append(y_i)
            xenew = ax[0] + ax[1]*xe + ax[2]*xe*xe + ax[3]*xe*ye + ax[4]*ye + ax[5]*ye*ye
            yenew = ay[0] + ay[1]*xe + ay[2]*xe*xe + ay[3]*xe*ye + ay[4]*ye + ay[5]*ye*ye

            # Update the bounds
            xmin = min(xmin,x[i])
            ymin = min(ymin,y[i])
            xmax = max(xmax,x[i])
            ymax = max(ymax,y[i])

            # Does the series tend to infinity
            if xmin < -1e10 or ymin < -1e10 or xmax > 1e10 or ymax > 1e10:
                drawit = False
                #print("infinite attractor")
                break

            # Does the series tend to a point
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                drawit = False
                #print("point attractor")
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

        # Classify the series according to lyapunov
        
        if drawit:
            if abs(lyapunov) < 10:
                print("neutrally stable")
                drawit = False
            elif lyapunov < 0:
                print(f"periodic {lyapunov} ")
                drawit = False 
            else:
                print(f"chaotic {lyapunov} ")

        # Save the image
        if drawit:
            saveAttractor(file_index,out_path, None, ax,ay,xmin,xmax,ymin,ymax,x[0],y[0], lyapunov)
            drawAttractor(file_index, out_path, x, y, xmin, xmax, ymin, ymax, MAXITERATIONS, 500, 500)
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

    with open(f"{path}/{name}.json", "w") as f:
        json.dump(parameters, f, indent=4)


def loadAttractor(pathname):
    with open(f"{pathname}", "r") as f:
        parameters = json.load(f)
    return parameters

def generateAttractorFromParameters(params, iters=MAXITERATIONS):
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

def drawAttractor(name, out_path, x, y, xmin, xmax, ymin, ymax, iters, width, height, pad_w=0, pad_h=0):

    # Save the image
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    
    for i in range(iters):
        # ix iy use the min and max to scale the points to the image size
        ix = width * (x[i] - xmin) / (xmax - xmin)
        iy = height * (y[i] - ymin) / (ymax - ymin)
        # TODO add padding to sides (just blank space at the limits)
        # TODO color based on iteration number, interpolation between two colors
        if i > 100:
            draw.point([ix, iy], fill="black")

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    image.save(f"{out_path}/{name}.png", "PNG")
    print(f"saved attractor to ./{out_path}/{name}.png")


if __name__ == "__main__":
    print("Starting...")
    createAttractor()
    # for filename in glob.glob("output/*.json"):
    #     print(f"Processing {filename}")
    #     params = loadAttractor(filename)
    #     attr = generateAttractorFromParameters(params, 500_000)
    #     name = os.path.basename(filename).split('.')[0]
    #     drawAttractor(name, "load_test_bigger", *attr, 1000, 1000)
    