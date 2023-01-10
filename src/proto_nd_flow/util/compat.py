
def assert_compat_version(version, other_version):
    '''
        Raises an AssertionError if the requested version (``version``) is
        not compatible with ``other_version``. Compatibility requires:

         - major versions are eqal
         - minor version of ``version`` is less than or equal to ``other_version``

        :param version: version ``str`` formatted ``'major.minor.subminor'``

        :param other_version: version ``str`` formatted ``'major.minor.subminor'``

        :returns: ``None``
    '''
    version = version.split('.')
    other_version = other_version.split('.')

    assert version[0] == other_version[0], f'Major version incompatible! Requires {version[0]} == {other_version[0]}'
    assert int(version[1]) <= int(other_version[1]), f'Minor version incompatible! Requires {version[0]}.{version[1]} <= {other_version[0]}.{other_version[1]}'
