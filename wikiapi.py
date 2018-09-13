#coding: utf-8
from urllib.parse import quote_plus
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd

def remove_tags(html):
    html = str(html)

    atag_pattern = r"<a.*</a>"
    html = re.sub(atag_pattern, "", html)

    imgtag_pattern = r"<p id=.*<img alt.*/></p>"
    html = re.sub(imgtag_pattern, "", html)

    divtag_pattern = r'<div aria-describedby="modal1Desc".*</div>'
    html = re.sub(divtag_pattern, "", html)
    return html

def get_fish_info_fromweb(fish_name, is_only_basic_info=True):
    fish_name = quote_plus(fish_name, encoding='utf-8')

    url = 'https://www.zukan-bouz.com/syu/'+(fish_name)
    html = urlopen(url)
    soup = BeautifulSoup(html)

    # select exp3 table
    table = soup.findAll('table', {'class': 'exp3'})[0]
    if not is_only_basic_info:
        return remove_tags(table)

    # select basic info tr
    basic_info = table.findAll('tr')[2]
    return remove_tags(basic_info)

def return_database_info(fish_name):
    filename = 'fish_poison_database/fish_poison_database.csv'
    database = pd.read_csv(filename, encoding='utf_8')
    database_exist=database[database['fish_name']==fish_name]
    fish_id=database_exist.iat[0,0]
    return fish_id,database_exist

def get_fish_info_from_csv(fish_name,fish_info_dict_bool,fish_info_dict_name):
    label, fish_data = return_database_info(fish_name)
    fish_info_str = ''
    keys = [k for k, v in fish_info_dict_bool.items() if v == True]
    fish_info_str=fish_info_str+f'<table border="1">'
    for key in keys:
        # print(fish_info_dict_name[key]+':'+fish_data.loc[label,key])
        fish_info_str = fish_info_str + f'<tr><td>{fish_info_dict_name[key]}</td><td>{fish_data.loc[label,key]}</td></tr>'
    fish_info_str=fish_info_str+f'</table>'
    return fish_info_str


if __name__ == '__main__':
    is_only_basic_info = False
    fish_name = "ソウシハギ"
    """
    get_fish_info_from_csv
    csvデータより情報を頂く
    引数：fish_name(魚の名前),fish_info_dict_bool(出力する情報（ブール値）),fish_info_dict_name(出力する情報のタグ（いじる必要なし）)
    戻り値：fish_info_str(出力する必要の文字列)
    """
    fish_info_key_name = ['fish_name', 'poison_info', 'habitat', 'season', 'cooking', 'note']
    fish_info_name = ['魚名', '毒情報', '生息域', '旬', '料理', '備考']
    fish_info_bool = [True, True, True, True, True, True]
    fish_info_dict_bool = dict(zip(fish_info_key_name, fish_info_bool))
    fish_info_dict_name = dict(zip(fish_info_key_name, fish_info_name))
    fish_info_str=get_fish_info_from_csv(fish_name,fish_info_dict_bool,fish_info_dict_name)
    print(fish_info_str)
    print('----------------')
    """
    get_fish_info_fromweb
    webより情報を頂く
    引数：fish_name(魚の名前),is_only_basic_info(基本情報だけかテーブル情報か,true:基本情報だけ)
    戻り値：fish_info(魚の情報)
    """
    fish_info = get_fish_info_fromweb(fish_name, is_only_basic_info)
    print("## fish infomation after removing")
    print(fish_info)
