from setuptools import setup, find_packages

setup(
    name="weatherforecasting_project",
    version="0.1.0",
    description="Autoregressive forecasting the weather in Stockholm",
    author="Vincent Dahlberg",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "pandas",
        "xarray",
        "seaborn",
        "scikit-learn",
        "cdsapi",
        "pyarrow",
        "tqdm",
        "PyYAML",
        
    ],
    python_requires=">=3.9"
)