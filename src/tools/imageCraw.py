import urllib.request
import re
import os
import urllib


def get_html(url):
    page = urllib.request.urlopen(url)
    html_a = page.read()
    return html_a.decode('utf-8')


def get_img(html):
    reg = r'https://[^\s]*?\.jpg'
    imgre = re.compile(reg)  # 转换成一个正则对象
    imglist = imgre.findall(html)  # 表示在整个网页过滤出所有图片的地址，放在imgList中
    x = 0        # 声明一个变量赋值
    path = 'resources/images/'  # 设置图片的保存地址
    if not os.path.isdir(path):
        os.makedirs(path)  # 判断没有此路径则创建
    paths = path + '\\'  # 保存在test路径下
    for imgurl in imglist:
        urllib.request.urlretrieve(imgurl, '{0}{1}.jpg'.format(paths, x))  # 打开imgList,下载图片到本地
        x = x + 1
        print('已经爬取{}张图片'.format(x))
    return imglist


URL="http://www.polayoutu.com/collections?utm_source=bigezhang.com&tdsourcetag=s_pcqq_aiomsg"
html_b = get_html(URL)  # 获取该网页的详细信息
get_img(html_b)  # 从网页源代码中分析下载保存图片
print("DONE!")