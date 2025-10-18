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
    
def randomTriadColors(lbound=0, ubound=255, three_colors=False):
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

def create_linear_colormap(colors=None, positions=None, name='custom_cmap', preset=None, 
                           recolor_base=None) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom linear colormap or use a preset colormap from matplotlib.
    The positions must sum up to 1.0 and allow to define where each color is placed in the gradient.
    This is useful to create smooth color gradients for rendering attractors using plt.imshow.

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


if __name__ == "__main__":
    # test the create_linear_colormap function
    
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # you wont see it but the first color is black by default
    cmap = create_linear_colormap(colors=[(255, 0, 0), (0, 255, 0)])
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')
    plt.savefig("test_colormap.png", bbox_inches='tight', pad_inches=0)

    cmap2 = create_linear_colormap(preset='viridis', recolor_base={'n':10, 'color':(0,0,0)})
    plt.imshow(gradient, aspect='auto', cmap=cmap2)
    plt.axis('off')
    plt.savefig("test_colormap2.png", bbox_inches='tight', pad_inches=0)
