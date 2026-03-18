from setuptools import setup, find_packages

setup(
    name="sam3_building_identifier",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # All heavy deps are expected to already exist in the geoai_sam conda env.
        # Listed here for documentation; pip install is not required.
        "samgeo>=1.0.1",
        "torch",
        "geoai",
        "rasterio",
        "shapely>=2.0",
        "numpy",
        "Pillow",
        "opencv-python",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "sam3-identify=sam3_building_identifier.__main__:main",
        ],
    },
)
