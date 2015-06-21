from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='metacritic',
      version='0.314',
      description='Reverse engineer the relative weights metacritic assigns to critics',
      url='http://github.com/shashank025/metacritic-weights',
      author='Shashank Ramaprasad',
      author_email='shashank.ramaprasad+metacritic@gmail.com',
      license='MIT',
      packages=['metacritic'],
      scripts=[
          'bin/mc_ytheta',
          'bin/mc_extract_raw_ratings',
          'bin/mc_extract_significant_critics',
          'bin/mc_prune',
          'bin/mc_partition',
          'bin/mc_train',
          'bin/mc_report_weights',
          'bin/mc_predict',
          'bin/mc_perf_report',
      ],
      install_requires=[
          'lxml',
          'numpy',
          'scipy',
        ],
      include_package_data=True,
      zip_safe=False)
