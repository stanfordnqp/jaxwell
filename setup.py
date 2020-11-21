import setuptools

setuptools.setup(
    name='jaxwell',
    version='0.1',
    description='3D iterative FDFD electromagnetic solver.',
    author='Jesse Lu',
    author_email='mr.jesselu@gmail.com',
    python_requires='>=3.6',
    install_requires=["numpy>=1.18.5", "jax>=0.2.4"],
    url='https://github.com/stanfordnqp/jaxwell',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
)
