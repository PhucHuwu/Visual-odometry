from setuptools import setup, find_packages

setup(
    name="visual-odometry",
    version="0.1.0",
    description="Visual Odometry system with multi-algorithm support and 3D trajectory visualization",
    author="Visual Odometry Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vo=main:main",
        ],
    },
)
