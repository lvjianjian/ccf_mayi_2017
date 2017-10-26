# coding = utf-8
import pandas  as pd


class shop_info(object):
    # file_name :the path of information.csv of shop
    def __init__(self, f=
    "../../data/train-ccf_first_round_shop_info.csv"):
        data = pd.read_csv(f)
        # drop duplicates
        self.data = data.drop_duplicates(["shop_id"])
        self.data.columns = ['shop_id', 'category_id', 'shop_longitude', 'shop_latitude', 'price', 'mall_id']

# if __name__ == "__main__":
#     shop_inf = shop_info()
