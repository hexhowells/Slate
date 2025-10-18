from setuptools import setup, find_packages

setup(
    name='Slate',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'slate': ['static/*']
    },
    install_requires=[
        'Flask>=2.0',
        'Flask-SocketIO>=5.3.0',
        'websockets>=10.0',
        'gym>=0.21.0',
        'opencv-python>=4.5.0',
        'numpy>=1.18.0',
		'websockets>=15.0.1'
    ],
    entry_points={
        'console_scripts': [
            'slate=slate.server:Slate'
        ],
    },
)
