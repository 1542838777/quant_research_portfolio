from setuptools import setup, find_packages

setup(
    name='quant_lib',
    version='0.1.0',
    description='量化研究框架',
    author='Quant Researcher',
    author_email='1542838771@qq.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'pyarrow>=6.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'tushare>=1.2.0',
        'pyyaml>=6.0',
        'joblib>=1.1.0',
        'pytest>=6.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)