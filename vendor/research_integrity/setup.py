from setuptools import setup, find_packages

setup(
    name="research_integrity",
    version="0.1.0",
    description="Shared research-integrity rails (cross-OOS battery, write-once holdout, FDR-aware "
                "promote bar, deployment-sanity) for portfolio-wide edge-search validation.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "scipy"],
)
