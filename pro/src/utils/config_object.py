# encoding = utf-8
import yaml


class Config_object(object):
    def __init__(self, config):
        stream = open(config, "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                if k != "train":
                    continue
                else:
                    for k1, v1 in v.items():
                        cmd = "self." + k + "=" + repr(v)
                        exec (cmd)
        stream.close()
