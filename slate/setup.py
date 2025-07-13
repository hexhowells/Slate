from setuptools import setup, find_packages

setup(
    name='ZyDash',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask>=2.0',
        'gym>=0.21.0',
        'opencv-python>=4.5.0',
        'numpy>=1.18.0'
    ],
    entry_points={
        'console_scripts': [
            'zydash=zydash.server:ZyDash'
        ],
    },
)
