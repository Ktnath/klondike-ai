from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="klondike-ai",
    version="0.1.0",
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "klondike_core",
            "core/Cargo.toml",
            binding=Binding.PyO3
        ),
    ],
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.11.0",
        "tqdm>=4.64.0",
        "colored>=1.4.3",
    ],
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
    author="Klondike AI Team",
    author_email="contact@klondike-ai.org",
    description="Une implémentation de Klondike Solitaire avec apprentissage par renforcement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Rust",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
