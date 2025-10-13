"""Manage colors and their interpolations for the attractors."""

import math
import random

def colorInterpolate(color1, color2, t, mode='linear') -> tuple[int, int, int]:
    """Interpolate between two colors.
    Usually the best interpolation mode is exponential, since:
    - Attractors tend to concentrate points in some areas, leaving other areas sparse. This means that
        initial non-dense areas will be coloured heavily with the first colors, and dense areas will be coloured
        heavily with the last colors.
        
    - Human vision is logarithmic in nature, so exponential interpolation often looks more natural.
    color1, color2: tuples of (R, G, B) with values from 0 to 255
    t: float from 0.0 to 1.0 indicating interpolation factor
    mode: interpolation mode, one of 'linear', 'log', 'exp', 'cos', 'bicubic'
    """

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