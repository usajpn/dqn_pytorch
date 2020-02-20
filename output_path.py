import os


class OutputPath(object):
    def __init__(self, path='./tmp.output/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.path = path

    def get_filepath(self, name):
        return os.path.join(self.path, name)


_default_output_path = None


def default_output_path():
    global _default_output_path
    if _default_output_path is None:
        _default_output_path = OutputPath()
    return _default_output_path
