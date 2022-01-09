###
# Exceptions
###
class FileTreeException(Exception):
    pass


class InvalidRootDirException(FileTreeException):
    pass


class InvalidTreeOperationException(FileTreeException):
    pass
