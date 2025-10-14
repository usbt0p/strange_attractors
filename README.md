# Strange attractors
From an article i found from https://paulbourke.net/fractals/lyapunov/

Automatic generation of attractors by using Lyapunov exponents to calculate promising chaothic attractors (and discarding point or infinity attractors)

# Some examples

| | | |
|-----------|-----------|-----------|
| <img src="examples/0.png" width="400"/> | <img src="examples/7.png" width="400"/> | <img src="examples/21.png" width="400"/> |
| <img src="examples/24.png" width="400"/> | <img src="examples/29.png" width="400"/> | <img src="examples/33.png" width="400"/> |
| <img src="examples/59.png" width="400"/> | <img src="examples/143_0.png" width="400"/> | <img src="examples/75.png" width="400"/> |
| <img src="examples/79.png" width="400"/> | <img src="examples/100.png" width="400"/> | <img src="examples/125.png" width="400"/> |
| <img src="examples/135.png" width="400"/> | <img src="examples/136.png" width="400"/> | <img src="examples/147.png" width="400"/> |

## TODO Some ideas to try 
- [ ] Add symbolic equations and save them as parameters. That way, automatic equation generation will be a possibility (with arbitrary params).
- [ ] Merge this into the other code to allow for colouring, etc.
- [ ] Test the correct loading with seed points: all attractors should work when loaded.
- [ ] Change the quadratic equation to use `a**x` instead of `a*x*x`.
- [ ] Test the Lyapunov normalization `lyapunov /= num_iters`.
- [ ] Implement fractal dimension calculation.
- [ ] Perform data analysis on a big generated set of attractors + their parameters:
    - [ ] Augment parameters and calculate stats.
    - [ ] Cluster by attractor features.
    - [ ] Find out the relation the non-interesting generated attractors have with pixel density / coefficients.
- [ ] Make higher dimensional attractors and project them into a smaller space.
- [ ] Make 3D attractors and plot them in 3D.
- [ ] Make a video / gif of the points appearing with their respective colour.
- [ ] add padding to sides of drawings (just blank space at the limits)
- [ ] add transparency to the histogram mode
- [ ] add `exists_ok=True` for ignoring existing files

## References!
https://www.nathanselikoff.com/training/tutorial-strange-attractors-in-c-and-opengl
https://www3.fi.mdp.edu.ar/fc3/SisDin2009/Clase%202/nousado/CHAOSCOPE/help/en/manual/attractors.htm
http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/
https://paulbourke.net/fractals/lyapunov/