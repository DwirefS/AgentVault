"""
AgentVault™ - Enterprise AI Agent Storage Platform on Azure NetApp Files
Setup configuration for the world's first enterprise-grade AI agent storage solution.

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def read_readme():
    """Read README for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AgentVault™ - Enterprise AI Agent Storage Platform on Azure NetApp Files"

setup(
    name="agentvault",
    version="1.0.0-alpha",
    description="Enterprise AI Agent Storage Platform on Azure NetApp Files",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Dwiref Sharma",
    author_email="DwirefS@SapientEdge.io",
    url="https://github.com/DwirefS/AgentVault",
    license="MIT",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.9",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
        "enterprise": [
            "redis-enterprise>=0.1.0",
            "azure-identity-broker>=1.0.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "agentvault=agentvault.cli:main",
            "agentvault-deploy=agentvault.deployment.cli:main",
            "agentvault-monitor=agentvault.monitoring.cli:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Networking",
        "Topic :: Database :: Database Engines/Servers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords=[
        "ai", "agents", "storage", "azure", "netapp", "enterprise", 
        "langchain", "autogen", "vector-database", "memory", "cache"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/DwirefS/AgentVault/issues",
        "Funding": "https://github.com/sponsors/DwirefS",
        "Source": "https://github.com/DwirefS/AgentVault",
        "Documentation": "https://agentvault.readthedocs.io/",
    },
    
    package_data={
        "agentvault": [
            "configs/*.yaml",
            "configs/*.json",
            "schemas/*.json",
            "templates/*.tf",
            "templates/*.yaml",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)