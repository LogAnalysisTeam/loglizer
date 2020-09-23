from setuptools import find_packages, setup

setup(
    name='loglizer',
    packages=find_packages(),
    version='0.1.0',
    description='loglizer: A log analysis toolkit for automated anomaly detection, extended',
    author='LOGPAI, Log Analysis Team, AIC, FEE CTU in Prague',
    license='MIT',
    install_requires=['sklearn', 'pandas'],
)
