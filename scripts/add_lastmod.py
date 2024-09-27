"""Add lastmod (with the file creation date) to all posts that don't have it."""

import os
import datetime

POST_FOLDER = os.path.join(os.path.dirname(__file__), "../content/posts")
for a, b, c in os.walk(POST_FOLDER):
    for file in c:
        path = os.path.join(a, file)
        lines = open(path, "r", encoding="utf-8").readlines()
        if any((x.startswith("lastmod:") for x in lines)):
            print(f"[*] {file} already has lastmod. updating.")
        else:
            lines.insert(2)
        stat = os.stat(path)
        lastmod = (
            datetime.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
        )
        print(f"[-] {file} : {lastmod}")
        lines[2] = f"lastmod: {lastmod}\n"
        open(path, "w", encoding="utf-8").writelines(lines)
