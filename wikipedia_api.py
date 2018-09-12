import lxml.html
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings()
http = urllib3.PoolManager()
r = http.request('GET', 'https://ja.wikipedia.org/w/api.php?format=xml&action=query&prop=info&titles=%E3%82%A8%E3%83%9E%E3%83%BB%E3%83%AF%E3%83%88%E3%82%BD%E3%83%B3')
html = lxml.html.fromstring(r.data.decode('utf-8'))
title = html.xpath('query/pages/page/@title')[0]

print(title)
print('--------------')


html = r.data.decode("utf-8")
print(html)
print('--------------')
soup = BeautifulSoup(html, "lxml")
print(soup.text)
