import re,requests,sys,os

file = open("links.txt")
download_path = r"M:\Course stuff\Fall 17\Masters project\ASPRI\CAIDA\series-1\raw data"
links = []
names = []
write_file = open("dl.txt","w")
for i in file:
	fname = re.compile(re.escape('">')+'.*').sub('',i)
	names.append(fname)
	links.append("http://data.caida.org/datasets/as-relationships/serial-1/"+fname)
	write_file.write("http://data.caida.org/datasets/as-relationships/serial-1/"+fname)
#print(len(links))
t = open("dl.txt","r")
os.chdir(download_path)

import gzip,urllib.request
for i in range(len(links)):
	url = links[i].strip()
	file_name = names[i].strip()
	r = requests.get(url, allow_redirects=True)
	#open(links[i], 'wb').write(r.content)
	open(names[i].strip(),"wb").write(r.content)
	#download(url,file_name)
