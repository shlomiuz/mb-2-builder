import re
from operator import methodcaller

PROPERTY_LINE = re.compile(r'^(?P<key>.*?)=(?P<value>.*?)$')


def _strip_surrounding_quotes(s):
    """
    Strip the surrounding quotes if they exist
    
    :param str s: 
    :return: the passed string without the surround quotes or the same string if they don't
    :rtype: str
    """
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    else:
        return s


def _from_stream(stream, lower_values):
    props = dict()
    for line_index, line in enumerate(map(methodcaller('strip'), stream)):
        if line == '': continue # ignore empty lines

        line_match = PROPERTY_LINE.match(line)
        if line_match is None:
            raise Exception('Failed parsing properties on line {}'.format(line_index + 1))

        key = line_match.group('key')
        value = line_match.group('value')

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
            value = list(map(_strip_surrounding_quotes, value.split(',')))
            if lower_values:
                value = list(map(methodcaller('lower'), value))
        else:
            value = _strip_surrounding_quotes(value)
            if lower_values:
                value = value.lower()

        props[key] = value if value != '' else None # Empty value is the same as having no value

    return props


def from_file(file_path, lower_values=False):
    """
    Load a properties file as dictionary
    
    :param str file_path: path to the file containing parameters
    :param bool lower_values: should values be lower cased
    :return: dictionary of the loaded parameters
    :rtype: dict
    """
    with open(file_path, mode='r') as f:
       return _from_stream(f, lower_values=lower_values)

