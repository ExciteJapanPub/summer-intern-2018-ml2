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

def get_fish_info_from_csv(fish_name):
    csv_filename = 'fish_poison_database/fish_poison_database.csv'
    database = pd.read_csv(csv_filename, encoding='utf_8')
    database_exist = database[database['fish_name'] == fish_name]
    fish_id = database_exist.iat[0, 0]
    #print('{0}:{1}'.format(database_exist.loc[fish_id, 'fish_name'], database_exist.loc[fish_id, 'poison_info']))
    return fish_id,database_exist


if __name__ == '__main__':
    is_only_basic_info = False
    fish_name = "ソウシハギ"
    """
    get_fish_info_from_csv
    csvデータより情報を頂く
    引数：fish_name(魚の名前)
    戻り値：label(魚ID),fish_data_csv(該当する魚のデータ)
    """
    label,fish_data_csv=get_fish_info_from_csv(fish_name)
    print('{0}:{1}'.format(fish_data_csv.loc[label, 'fish_name'], fish_data_csv.loc[label, 'poison_info']))
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
