import os
import glob
import re

def process_file(jax_path):
    mlx_path = jax_path.replace('_JAX.py', '_MLX.py')
    with open(jax_path, 'r') as f:
        content = f.read()

    # Replace generator trace
    content = content.replace('tools/generate_jax_variants.py', 'tools/generate_mlx_variants.py')
    
    # Replace JAX imports with MLX imports
    content = content.replace('import jax.numpy as np', 'import mlx.core as mx\n    import numpy as np')
    content = content.replace('_HAS_JAX', '_HAS_MLX')
    
    # Simple search and replace for JAX suffixed classes
    content = content.replace('_JAX =', '_MLX =')
    
    # Replace JAX-specific things
    content = content.replace('CrowdingDiversity_JAX', 'CrowdingDiversity_MLX')
    content = content.replace('FunctionalDiversity_JAX', 'FunctionalDiversity_MLX')
    content = content.replace('FuncionalDiversityMNN_JAX', 'FuncionalDiversityMNN_MLX')
    
    with open(mlx_path, 'w') as f:
        f.write(content)
        
    print(f"Generated {mlx_path}")

if __name__ == '__main__':
    # Find all JAX variants
    for jax_file in glob.glob('operators/**/*_JAX.py', recursive=True):
        if 'metrics_JAX.py' in jax_file:
            process_file(jax_file)
