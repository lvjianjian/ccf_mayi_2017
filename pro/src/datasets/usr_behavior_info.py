# coding = utf-8 
import pandas as pd
class usr_behavior_info(object):
	def __init__(self, f = "../../data/train-ccf_first_round_user_shop_behavior.csv"):
		self.data = pd.read_csv(f)
		
