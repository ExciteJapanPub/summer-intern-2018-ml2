#coding: utf-8
from urllib.parse import quote_plus
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

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

if __name__ == '__main__':
    is_only_basic_info = False
    fish_name = "ソウシハギ"
    fish_info = get_fish_info_fromweb(fish_name, is_only_basic_info)
    print("## fish infomation after removing")
    print(fish_info)
