import sys

def err(msg):
    print("error: " + msg, file=sys.stderr)


def debug(msg):
    print("debug: " + msg, file=sys.stderr)


def warning(msg):
    print("warning: " + msg, file=sys.stderr)


def info(msg):
    print("info: " + msg, file=sys.stderr)
