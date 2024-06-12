'''Add lastmod (with the file creation date) to all posts that don't have it.'''
import os
import datetime
POST_FOLDER = '../content/posts'
for file in os.listdir(POST_FOLDER):
    path = os.path.join(POST_FOLDER, file)
    lines = open(path, 'r', encoding='utf-8').readlines()
    if any((x.startswith('lastmod:') for x in lines)):
        print(f'[!] {file} already has lastmod')
        continue
    stat = os.stat(path)
    lastmod = datetime.datetime.fromtimestamp(stat.st_ctime).astimezone().isoformat()
    print(f'[-] {file} : {lastmod}')
    lines.insert(2, f'lastmod: {lastmod}\n')
    open(path,'w', encoding='utf-8').writelines(lines)
