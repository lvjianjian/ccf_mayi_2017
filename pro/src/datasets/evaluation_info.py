# encoding = utf-8 
import pandas as pd
class evaluation_info(object):
	def __init__(self, f = "../../data/AB-test-evaluation_public.csv"):
		self.data = pd.read_csv(f)

