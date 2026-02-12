"""Manage colors and their interpolations for the attractors."""

import math
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def colorInterpolate(color1, color2, t, mode='linear') -> tuple[int, int, int]:
    """Interpolate between two colors.
    Usually the best interpolation mode is exponential, since human vision is logarithmic.
    This interpolation mode however is worse than the one used in the histogram coloring, and is not used there.
    mode: interpolation mode, one of 'linear', 'log', 'exp', 'cos', 'bicubic'
    """
    match mode:
        case 'lin': pass  # t is already linear
        case 'log': t = math.log(t * (math.e - 1) + 1)  
        case 'exp': t = (math.exp(t) - 1) / (math.e - 1)
        case 'cos': t = (1 - math.cos(math.pi * t)) / 2 
        case 'bicubic': t = t * t * (3 - 2 * t)  # smoothstep
        case _: pass  # default to linear if unknown mode

    r = int(color1[0] + t * (color2[0] - color1[0]))
    g = int(color1[1] + t * (color2[1] - color1[1]))
    b = int(color1[2] + t * (color2[2] - color1[2]))
    
    return (r, g, b)

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
    
def randomTriadColors(lbound=0, ubound=255, three_colors=True):
    '''Options allow to return either three or two triadic colors.'''
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

    if three_colors:
        return (r, g, b), (triad1_r, triad1_g, triad1_b), (triad2_r, triad2_g, triad2_b)
    # coin toss for the second color
    if random.random() < 0.5:
        triad = (triad1_r, triad1_g, triad1_b)
    else:
        triad = (triad2_r, triad2_g, triad2_b)

    return (r, g, b), triad

def randomColoursGoldenRatio(lbound=0, ubound=255, n=2):
    """Generate n colors based on the golden ratio to ensure good distribution.
    Returns a list of n colors as (R, G, B) tuples.
    """
    assert n >= 2, "n must be at least 2"

    # Generate a random base color within bounds
    r = random.randint(lbound, ubound)
    g = random.randint(lbound, ubound)
    b = random.randint(lbound, ubound)

    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    span = ubound - lbound + 1

    colors = [(r, g, b)]
    for _ in range(n - 1):
        # Calculate next color using golden ratio, mod makes sure we stay within bounds
        golden_r = lbound + int(((r - lbound) * phi) % span)
        golden_g = lbound + int(((g - lbound) * phi) % span)
        golden_b = lbound + int(((b - lbound) * phi) % span)
        
        colors.append((golden_r, golden_g, golden_b))
        r, g, b = golden_r, golden_g, golden_b

    return colors

def recolor_colormap_base(cmap, n=1, color=(0, 0, 0), resample_size=256) -> mcolors.ListedColormap:
    """
    Modify the first `n` entries of a colormap to a specific RGB color.
    Replacing the first `n` colors is useful to set the background color of an attractor.
    """
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    # Convert color to 0-1 range if necessary
    if max(color) > 1:
        color = tuple(c / 255 for c in color)
    color = np.array((*color, 1.0))  # asegurar canal alfa

    # Resample the colormap to a certain number of colors
    newcolors = cmap(np.linspace(0, 1, resample_size))
    newcolors[:n, :] = color
    return mcolors.ListedColormap(newcolors)

def create_linear_colormap(colors: list[tuple]=None, 
                           positions: list[float]=None, 
                           name='custom_cmap', 
                           preset: str=None, 
                           recolor_base : dict = None #{'n':1, 'color':(0,0,0)}
                           ) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom linear colormap or use a preset colormap from matplotlib.
    The positions must sum up to 1.0 and allow to define where each color is placed in the gradient.
    This is useful to create smooth color gradients for rendering attractors using plt.imshow.
    
    Recolor base allows to modify the first n colors of the colormap to a specific color.
    For example: `{'n':1, 'color':(0,0,0), resample_size:256}` will set the first color to black.

    https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#creating-listed-colormaps
    """
    # ensure colors are rgb in 0-255 and then convert to 0-1
    if preset:
        if recolor_base:
            return recolor_colormap_base(plt.get_cmap(preset), **recolor_base)
        else:
            return plt.get_cmap(preset)
    
    if colors is None and preset is None: # default colormap
        if recolor_base:
            return recolor_colormap_base(plt.get_cmap('magma'), **recolor_base)
        else:
            return plt.get_cmap('magma')  
    
    if colors:
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors), "Colors must be tuples of (R, G, B)"
        colors = [(r/255, g/255, b/255) for r, g, b in colors]

    # If positions are not provided, distribute them evenly
    n = len(colors)
    if positions is None:
        positions = [i / (n - 1) for i in range(n)]

    if len(colors) != len(positions):
        raise ValueError("Colors and positions must have the same length")
    
    finalcmap = mcolors.LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
    if recolor_base:
        finalcmap = recolor_colormap_base(finalcmap, **recolor_base)
    return finalcmap

# from https://nicoguaro.github.io/posts/cyclic_colormaps/, slightly
# modified to match my needs. the idea of using circles in 3d space is great
def randomCyclicColormap(number, plot_results=False, deterministic=False):
    '''Create `number` cyclic colormaps (a cyclic colormap starts and 
    ends in the same color). Optionally plot results (in `.svg` and `.png`).'''
    
    if plot_results:
        # to plot in a square grid, we'll want to find the nearest bigger square
        nbsq = int(math.ceil(number**0.5))
        nx, ny = nbsq, nbsq

        fig, fig2 = plt.figure(), plt.figure()
        azimuths, zeniths = np.arange(0, 361, 1), np.arange(30, 70, 1)
        values = azimuths * np.ones((len(zeniths), len(azimuths)))

    cmaps = []
    for cont in range(number):
        if deterministic: np.random.seed(seed=cont)

        # we are working on a 3d space that represents the RGB space
        mat = np.random.rand(3, 3)
        # this is a matrix factorization. rot_mat is upper triang. and allows us to rotate in Z later
        # https://math.stackexchange.com/questions/1141763/qr-decomposition-interpretation
        rot_mat, _ = np.linalg.qr(mat)
        
        # set a random radius and center, ensure center doesn't go out of bounds from out cube
        radius = np.random.uniform(0.1, 0.5)
        center = np.random.uniform(radius, 1 - radius, size=(3, 1))
        #center = np.clip(center, 0, 1)

        # t is the angle in steps, x and y descirbe a circle in parametric coords in 2d
        t = np.linspace(0, 2*np.pi, 256)
        x = radius*np.cos(t)
        y = radius*np.sin(t)
        z = 0.0*np.cos(t) # this stays at zero
        # now we stack vectors and rotate. whitout rotation Z dimension would stay constant
        X = np.vstack((x, y, z))
        X = rot_mat.dot(X) + center

        # need transpose: (3, 256) -> (256, 3)
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', X.T) 
        cmaps.append(cmap)
        
        if plot_results:
            # plotting stuff
            ax = fig.add_subplot(ny, nx, 1 + cont, projection='polar') # polar makes plot a circle
            ax.pcolormesh(azimuths*np.pi/180.0, zeniths, values, cmap=cmap, shading='auto') # ngl, idk what this does
            ax.set_xticks([]); ax.set_yticks([])

            # yada yada matplotlib 3d
            ax2 = fig2.add_subplot(ny, nx, 1 + cont, projection='3d')
            ax2.plot(X[0, :], X[1, :], X[2, :])
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_zlim(0, 1)
            ax2.view_init(30, -60)
            ax2.set_xticks([0, 0.5, 1.0]); ax2.set_yticks([0, 0.5, 1.0]); ax2.set_zticks([0, 0.5, 1.0])
            ax2.set_xticklabels([]); ax2.set_yticklabels([]); ax2.set_zticklabels([])

    if plot_results:
        fig.savefig("random_cmaps.png", dpi=300, transparent=True)
        fig2.savefig("random_cmaps_traj.svg", transparent=True)
    
    return cmaps


if __name__ == "__main__":
    # test the create_linear_colormap function
    
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # you wont see it but the first color is black by default
    cmap = create_linear_colormap(colors=[(255, 0, 0), (0, 255, 0)])
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')
    plt.savefig("test_colormap.png", bbox_inches='tight', pad_inches=0)

    # now skew the colormap with positions
    cmap = create_linear_colormap(colors=[(0,0,0), (2, 24, 107), (255, 145, 0)])
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')
    plt.savefig("test_colormap_positions_1.png", bbox_inches='tight', pad_inches=0)

    positions = [0.0, 0.4, 0.8, 1.0]
    cmap = create_linear_colormap(colors=[(255,255,255), (2, 24, 107), (255, 145, 0), (255,255,255)], positions=positions)
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')
    plt.savefig("test_colormap_positions_2.png", bbox_inches='tight', pad_inches=0)

    cmap2 = create_linear_colormap(preset='viridis', recolor_base={'n':10, 'color':(0,0,0)})
    plt.imshow(gradient, aspect='auto', cmap=cmap2)
    plt.axis('off')
    plt.savefig("test_colormap2.png", bbox_inches='tight', pad_inches=0)

    # test the randomCyclicColormap func with two numbers of cmaps
    cmaps = randomCyclicColormap(7, plot_results=True)
    cmaps = randomCyclicColormap(16, plot_results=True)