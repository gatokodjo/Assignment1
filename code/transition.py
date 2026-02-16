# -*- coding: utf-8 -*-


class Transition(object):

    LEFT_ARC = "LEFTARC"
    RIGHT_ARC = "RIGHTARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self):
        raise ValueError("Do not construct this object!")

    @staticmethod
    def left_arc(conf, relation):
        if not conf.stack or not conf.buffer:
            return -1

        s = conf.stack.pop()
        b = conf.buffer[0]
        conf.arcs.append((b, relation, s))

    @staticmethod
    def right_arc(conf, relation):
        if not conf.buffer or not conf.stack:
            return -1

        s = conf.stack[-1]
        b = conf.buffer.pop(0)
        conf.stack.append(b)
        conf.arcs.append((s, relation, b))

    @staticmethod
    def reduce(conf):
        if not conf.stack:
            return -1

        conf.stack.pop()

    @staticmethod
    def shift(conf):
        if not conf.buffer:
            return -1

        b = conf.buffer.pop(0)
        conf.stack.append(b)
