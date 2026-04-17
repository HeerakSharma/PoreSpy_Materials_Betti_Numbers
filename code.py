import numpy as np
import porespy as ps
import gudhi
import pandas as pd
import tifffile
import os
from tqdm import tqdm  # Progress bar

num_materials = 20
shape = [200, 200, 200]
padding = 10

method = 'blobs'
# method = 'spheres'
# method = 'cylinders'

porosity = 0.1
blobiness = 1.2

radius = 10
maxiter = 100

theta_max = 90
phi_max = 90
length = 10

#Method blobs
def generate_material_blobs(shape, porosity, blobiness):
    material = ps.generators.blobs(
            shape=shape, 
            porosity=1-porosity, 
            blobiness=blobiness,
        )
    return ~material

#Method overlapping_spheres
def generate_material_spheres(shape, porosity, radius, maxiter):
    material = ps.generators.overlapping_spheres(
            shape=shape, 
            r=radius, #radius of spheres
            porosity=1-porosity,
            maxiter=maxiter
        )
    return ~material

#Method overlaping_cylinders
def generate_material_cylinders(shape, porosity, radius, length, phi_max, theta_max, maxiter):
    material = ps.generators.cylinders(
            shape=shape, 
            r=radius, #radius of cylinders
            length=length, #length of cylinders
            porosity=1-porosity, 
            phi_max=phi_max, 
            theta_max=theta_max,
            maxiter=maxiter
        )
    return ~material

# Takes input the method and return a material generated using that method with parameters specified in the above code cell
def generate_material(method, shape_ = shape, porosity_ = porosity, blobiness_ = blobiness, radius_ = radius, length_ = length, phi_max_ = phi_max, theta_max_ = theta_max, maxiter_ = maxiter):
    if method == 'blobs':
        material = generate_material_blobs(shape_, porosity_, blobiness_)
    
    elif method == 'spheres':
        material = generate_material_spheres(shape_, porosity_, radius_, maxiter_)
    
    elif method == 'cylinders':
        material = generate_material_cylinders(shape_, porosity_, radius_, length_, phi_max_, theta_max_, maxiter_)
    
    else:
        raise ValueError(f"Invalid method '{method}'. Expected one of: 'blobs', 'spheres', 'cylinders'")
        
    return material.astype(int)

# Centers input material by padding it with specified paddin value
def center_material_with_padding(material, padding):
    """
    Centers a 2D array in a larger canvas with specified padding on all sides.
    
    Parameters:
    - material: 2D numpy array
    - padding: int (pixels of space to add around the material)
    
    Returns:
    - centered_material: 2D numpy array with new dimensions
    """
    # np.pad takes a sequence of ((top, bottom), (left, right))
    # We apply the same padding to all sides to keep it centered
    centered_material = np.pad(
        material, 
        pad_width=padding, 
        mode='constant', 
        constant_values=0
    )
    
    return centered_material.astype(int)

def fix_connectivity(arr, subdivision_factors, lr_axis=-1):
    """Subdivides array with the P->Q rightmost edge rule."""
    if isinstance(subdivision_factors, int):
        subdivision_factors = (subdivision_factors,) * arr.ndim
        
    subdivided = arr.copy()
    for axis, factor in enumerate(subdivision_factors):
        subdivided = np.repeat(subdivided, factor, axis=axis)
        
    left_slices = [slice(None)] * arr.ndim
    right_slices = [slice(None)] * arr.ndim
    left_slices[lr_axis] = slice(None, -1)
    right_slices[lr_axis] = slice(1, None)
    
    condition = (arr[tuple(left_slices)] == 0) & (arr[tuple(right_slices)] == 1)
    
    pad_width = [(0, 0)] * arr.ndim
    pad_width[lr_axis] = (0, 1) 
    mask_P = np.pad(condition, pad_width, mode='constant', constant_values=False)
    
    mask_sub = mask_P.copy()
    for axis, factor in enumerate(subdivision_factors):
        mask_sub = np.repeat(mask_sub, factor, axis=axis)
        
    f_lr = subdivision_factors[lr_axis]
    rightmost_filter = np.zeros(subdivided.shape[lr_axis], dtype=bool)
    rightmost_filter[f_lr - 1 :: f_lr] = True 
    
    broadcast_shape = [1] * arr.ndim
    broadcast_shape[lr_axis] = -1
    rightmost_filter = rightmost_filter.reshape(broadcast_shape)
    
    final_mask = mask_sub & rightmost_filter
    subdivided[final_mask] = 1
    
    return subdivided

def get_betti_particles(image):
    """
    Computes Betti numbers (b0, b1, b2) for the particle space (value 1).
    
    Method:
    1. Invert the image (Particles become 0, Voids become 1).
    2. Create a Cubical Complex.
    3. Compute persistence.
    4. Count features that are born at <=0 and persist (die > 0).
    """
    # Invert: Particles (1->0), Voids (0->1)
    # This places particles in the first filtration level
    inverted_image = 1 - image
    
    cc = gudhi.CubicalComplex(
        dimensions=inverted_image.shape, 
        top_dimensional_cells=inverted_image.flatten() #this is the problem here: this forces 26 connectivity but i think 6 connectivity is the way to go
    )
    
    cc.compute_persistence()
    
    # Calculate Betti numbers for the sublevel set at 0
    # We look for intervals (birth, death) where birth <= 0 and death > 0
    betti_numbers = []
    for dim in range(3):
        intervals = cc.persistence_intervals_in_dimension(dim)
        count = 0
        for birth, death in intervals:
            if birth <= 0 and death > 0:
                count += 1
        betti_numbers.append(count)
        
    return betti_numbers


# Method B: Ab Initio
# Calculate directly on the image where Voids=0.
# Since Gudhi processes sublevel sets, and Voids are 0, we can just pass the raw material directly without the inversion step used in Cell 4.
def get_betti_voids(image):
    cc = gudhi.CubicalComplex(
        dimensions=image.shape, 
        top_dimensional_cells=image.flatten()
    )
    
    cc.compute_persistence()
    
    betti_numbers = []
    for dim in range(3):
        intervals = cc.persistence_intervals_in_dimension(dim)
        count = 0
        for birth, death in intervals:
            if birth <= 0 and death > 0:
                count += 1
        betti_numbers.append(count)
        
    return betti_numbers

def generate_data(method, shape, porosity, blobiness, radius, length, phi_max, theta_max, maxiter):
    # 1. Dynamic Folder Naming
    shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}"
    
    if method == 'blobs':
        param_str = f"blobiness={blobiness}"
    elif method == 'spheres':
        param_str = f"r={radius}_maxiter={maxiter}"
    elif method == 'cylinders':
        param_str = f"r={radius}_length={length}_phi={phi_max}_theta={theta_max}_maxiter={maxiter}"
    else:
        raise ValueError(f"Invalid method selected: {method}")

    # Create the specific subfolder name
    subfolder_name = f"{shape_str},{porosity},{method},{param_str}"
    
    # Combine with the 'materials' base folder
    # This creates: materials/100x100x100,0.65,blobs.../
    full_output_path = os.path.join('materials', subfolder_name)
    
    # Ensure both 'materials' and the subfolder exist
    os.makedirs(full_output_path, exist_ok=True)
    
    print(f"--- Processing Method: {method} ---")
    print(f"Saving to: {full_output_path}")
    print(f"Generating {num_materials} samples...")

    # 2. Indexing (Prevent Overwrite)
    existing_files = [f for f in os.listdir(full_output_path) if f.endswith('.tif')]
    start_idx = len(existing_files)
    
    data_records = []

    for i in tqdm(range(num_materials)):
        current_idx = start_idx + i
        file_name = f"sample_{current_idx:04d}.tif"
        
        # 3. Generate Material
        im = generate_material(
            method=method,
            shape_=shape,
            porosity_=porosity,
            blobiness_=blobiness,
            radius_=radius,
            length_=length,
            phi_max_=phi_max,
            theta_max_=theta_max,
            maxiter_=maxiter
        )
        
        # Ensure Boolean
        # im = im > 0

        # 4. Create Variations
        im_particle = im
        im_cent_particle = center_material_with_padding(im, padding)
        im_fixed_connectivity_ = fix_connectivity(im_cent_particle,2,lr_axis=1)
        im_fixed_connectivity = fix_connectivity(im_fixed_connectivity_,2,lr_axis=2)


        # 5. Compute Topology
        b_part = get_betti_particles(im_particle)
        b_void = get_betti_voids(im_particle)
        b_cent_part = get_betti_particles(im_cent_particle)
        b_cent_void = get_betti_voids(im_cent_particle)
        b_fixed_connectivity_part = get_betti_particles(im_fixed_connectivity)
        b_fixed_connectivity_void = get_betti_voids(im_fixed_connectivity)

        # 6. Save Image
        save_path1 = os.path.join(full_output_path, file_name)
        tifffile.imwrite(save_path1, (im_fixed_connectivity.astype(np.uint8) * 255))
        
        file_name2 = 'fixed_'+file_name
        save_path2 = os.path.join(full_output_path, file_name2)
        tifffile.imwrite(save_path2, (im_fixed_connectivity.astype(np.uint8) * 255))

        # 7. Log Data
        row = {
            'filename': file_name,
            'method': method,
            'target_porosity': porosity,
            'calc_porosity': ps.metrics.porosity(im),
            
            # Standard Stats
            'b0_part': b_part[0], 'b1_part': b_part[1], 'b2_part': b_part[2],
            'b0_void': b_void[0], 'b1_void': b_void[1], 'b2_void': b_void[2],
            
            # Centered/Padded Stats
            'b0_cent_part': b_cent_part[0], 'b1_cent_part': b_cent_part[1], 'b2_cent_part': b_cent_part[2],
            'b0_cent_void': b_cent_void[0], 'b1_cent_void': b_cent_void[1], 'b2_cent_void': b_cent_void[2],

            # Fixed Connectivity Stats
            'b0_fixed_part': b_fixed_connectivity_part[0], 'b1_fixed_part': b_fixed_connectivity_part[1], 'b2_fixed_part': b_fixed_connectivity_part[2],
            'b0_fixed_void': b_fixed_connectivity_void[0], 'b1_fixed_void': b_fixed_connectivity_void[1], 'b2_fixed_void': b_fixed_connectivity_void[2],
        }

        data_records.append(row)

    # 8. Save CSV inside the subfolder
    csv_path = os.path.join(full_output_path, 'betti_results.csv')
    df = pd.DataFrame(data_records)
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
    print(f"Done! Saved {num_materials} materials to '{full_output_path}'.")


num_materials = 10
side = 50
shape = [side, side, side]
padding = int(side/25)

# method = 'blobs'
# method = 'spheres'
method = 'cylinders'

porosity = 0.1
blobiness = 1.2

radius = 10
maxiter = 100

theta_max = 90
phi_max = 90
length = 10



por=0.1
while (por <=0.9):
    porosity = por
    r = 5
    while (r <= 25):
        radius = r
        l = 10
        while (l <= 25):
            generate_data(method, shape, porosity, blobiness, radius, length, phi_max, theta_max, maxiter)
            l = l + 5
        r = r + 5
    por = por + 0.1
