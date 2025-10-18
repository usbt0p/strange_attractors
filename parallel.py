
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import logging
from tqdm import tqdm
from lyapunov_exponents import loadAttractor, generateAttractorFromParameters, drawAttractor

def draw_single_attractor(filename, render_iterations=None, **kwargs):
    """Worker function to render a single attractor file
    from its parameters.
    This is designed to be as a worker function for multiprocessing, or
    can be used standalone."""
    
    if render_iterations is None:
        raise ValueError("render_iterations must be specified")

    try:
        logging.info(f"Processing {filename}")
        params = loadAttractor(filename)
        x, y, xmin, xmax, ymin, ymax = \
            generateAttractorFromParameters(params, render_iterations)
        logging.info("Correctly loaded parameters.")
        
        name = os.path.basename(filename).split('.')[0]
        drawAttractor(
                name, 
                xmin, xmax,
                ymin, ymax,
                x, y,
                maxiterations=render_iterations,
                **kwargs
            )
        logging.info(f"Finished rendering {filename}")
        return f"Successfully processed {filename}"
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        raise e
    
def draw_attractors_in_parallel(filenames, render_iterations, n_processes=cpu_count()-2, batch_size=100, **kwargs):
    """Render multiple attractor files in parallel using multiprocessing.
    Sets up a pool of worker processes and processes in batches to manage memory usage.
    """
    
    logging.info(f"Processing {len(filenames)} files using {n_processes} processes...")
    
    try:
        for i in range(0, len(filenames), batch_size):
            print(f"Batch 1: processing files {i+1} to {min(i + batch_size, len(filenames))}...")
            batch = filenames[i:i + batch_size]

            with Pool(processes=n_processes) as pool:
                # Use partial because the worker function needs extra positional args bersides keyword args
                worker_func = partial(draw_single_attractor, render_iterations=render_iterations, **kwargs)
                # Use imap for progress tracking
                results = list(tqdm(
                    # TODO change imap for map since its much faster, set smaller batch sizes
                    # and just use tqdm on the outer for loop 
                    pool.imap(worker_func, batch),
                    total=len(batch),
                    desc="Rendering..."
                ))
            # results summary
            successful = sum(1 for r in results if r.startswith("Successfully"))
            failed = len(results) - successful
            print(f"\tBatch {i//batch_size + 1}: {successful} successful, {failed} failed")
    
    except KeyboardInterrupt: # allow graceful exit on Ctrl+C
        print("Process interrupted by user. Exiting...") 
    
    return results