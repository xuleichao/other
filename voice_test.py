import urllib.request
from bs4 import BeautifulSoup
from aip import AipSpeech

def get_text(url):
    html = urllib.request.urlopen(url)
    source = html.read() 
    soup = BeautifulSoup(source,'lxml')
    p_tag = soup.find_all('p')
    text = []
    for i in p_tag:
        text.append(i.text)
    return text

def baidu_voice(text):
    APP_ID = '9892038'
    APP_KEY = 'pe4l1Wl4UBn6wzuNCRi41mw7'
    PSW  = 'FaHVA02B9K3fCLRGDugkiox0EeADolca' 
    
    aipSpeech = AipSpeech(APP_ID,APP_KEY,PSW)
    word_count = len(''.join(text))
    if word_count > 500:
        num = word_count // 500
        ys = word_count % 500
        for count in range(0,num):
            result = aipSpeech.synthesis(''.join(text)[count * 500:(count+1) * 500],'zh','1')
            if not isinstance(result,dict):
                with open('baidu' + str(count) + '.mp3','wb') as f:
                    f.write(result)
    else:
        result = aipSpeech.synthesis(text,'zh','1')
        if not isinstance(result,dict):
            with open('baidu.mp3','wb') as f:
                f.write(result)
text = get_text('http://mp.weixin.qq.com/s/p_wpy21ejHLruOB-7rejSw')
baidu_voice(text)
