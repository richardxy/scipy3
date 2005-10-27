
def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('scipy',parent_package,top_path)
    #config.add_subpackage('sandbox')
    config.add_subpackage('utils')
    config.add_subpackage('io')
    config.add_subpackage('fftpack')
    config.add_subpackage('signal')
    config.add_subpackage('integrate')
    config.add_subpackage('linalg')
    #config.add_subpackage('special')
    #config.add_subpackage('sparse')
    config.add_subpackage('optimize')
    config.add_subpackage('stats')
    config.add_subpackage('interpolate')
    config.add_subpackage('sparse')
    config.add_subpackage('cluster')
    config.make_svn_version_py()  # installs __svn_version__.py
    config.make_config_py('__scipy_config__')
    return config

if __name__ == '__main__':
    from scipy.distutils.core import setup
    setup(**configuration(top_path='').todict())
