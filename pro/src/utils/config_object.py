# encoding = utf-8
import yaml
class Config_object(object):
	def __init__(self, config):
		stream = open(config, "r")
		docs = yaml.load_all(stream)
		self.extractors = []
		for doc in docs:
			for k, v in doc.items():
				if k.startswith("extractor"):
					self.extractors.append(v)
		stream.close()				
