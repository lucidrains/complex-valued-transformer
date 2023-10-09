from setuptools import setup, find_packages

setup(
  name = 'complex-valued-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.8',
  license='MIT',
  description = 'Complex Valued Transformer / Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/complex-valued-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanisms',
    'transformers',
    'complex domain'
  ],
  install_requires=[
    'einops>=0.7.0',
    'torch>=1.12'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
