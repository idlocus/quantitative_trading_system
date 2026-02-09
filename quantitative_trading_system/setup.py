from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='quantitative_trading_system',
    version='0.1.0',
    description='A comprehensive quantitative trading system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/quantitative_trading_system',
    author='Your Name',
    author_email='your.email@example.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'qts=main:main',
        ],
    },
)