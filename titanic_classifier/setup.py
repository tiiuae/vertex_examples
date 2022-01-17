from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'xgboost==1.5.1',
    'tensorboardX==2.4',
    'loguru==0.5.3',
]

setup(
    name='trainer2',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='A Tutorial of Vertex AI'
)
