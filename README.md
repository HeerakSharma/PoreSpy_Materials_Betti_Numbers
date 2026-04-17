This repository contains the code (code.py) and materials that the code generated along with their Betti numbers (Downloads/materials.zip). This work is done as part of my MS thesis at IISER Pune with Prof. Vijay Natarajan.

Below is summary of the code generated using Gemini and verified to be correct my me.

## 1. Material Generation Methods
The code implements three distinct mathematical models to simulate different types of porous microstructures using the `PoreSpy` library:

* **Blobs:** Utilizes Gaussian random fields to create smooth, interconnected organic structures.
* **Spheres:** Simulates a particulate medium by randomly placing overlapping spheres until a target porosity is reached.
* **Cylinders:** Models fibrous materials by distributing rods with specific orientation constraints ($phi$ and $theta$ angles).



---

## 2. Geometric Preprocessing & Transformations
Before the topological analysis is performed, the script applies specific transformations to the generated 3D volumes:

### Centering and Padding
The `center_material_with_padding` function adds a "blank" border (zeros) around the material. This is a standard procedure in digital rock physics to ensure that the material is treated as a self-contained object, preventing the "boundary" of the simulation box from being counted as a topological feature.

### Connectivity Fixing
The `fix_connectivity` function is a specialized utility that upsamples the grid. It applies a "rightmost edge rule" to resolve ambiguities in voxel connectivity. In 3D imaging, the choice between **6-connectivity** (faces) and **26-connectivity** (faces, edges, and corners) can drastically change topological results; this function likely prepares the data for a consistent interpretation by the homology algorithm.

---

## 3. Topological Analysis (Betti Numbers)
The core scientific output of the script is the calculation of **Betti numbers** for both the **solid phase** (particles) and the **void phase** (pores) using the `Gudhi` library.

| Betti Number | Physical Interpretation |
| :--- | :--- |
| **$\beta_0$** | **Connected Components:** The number of isolated "islands" of material. |
| **$\beta_1$** | **Handles/Loops:** The number of redundant pathways or "donut holes" in the structure. |
| **$\beta_2$** | **Enclosed Cavities:** The number of bubbles or voids completely trapped within the solid. |

## 4. Automated Workflow
The script is structured to run as a high-throughput experiment runner:

1.  **Dynamic Directory Management:** It automatically creates folders named based on the simulation parameters (e.g., `materials/50x50x50,0.1,cylinders...`).
2.  **Parameter Sweeping:** The script uses nested `while` loops to iterate through ranges of:
    * **Porosity:** From 10% to 90%.
    * **Feature Size:** Varying radii and lengths for the inclusions.
3.  **Data Persistence:**
    * **TIFF Files:** Saves the 3D structures as image stacks for visual inspection.
    * **CSV Logging:** Appends all calculated Betti numbers and actual porosities to a centralized `betti_results.csv` for downstream statistical analysis.

---
