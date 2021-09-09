
def assert_compat_version(version, other_version):
    version = version.split('.')
    other_version = other_version.split('.')

    assert version[0] == other_version[0], f'Major version incompatible! Requires {version[0]} == {other_version[0]}'
    assert int(version[1]) >= int(other_version[1]), f'Minor version incompatible! Requires {version[0]}.{version[1]} >= {other_version[0]}.{other_version[1]}'
