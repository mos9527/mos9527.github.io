"""Add lastmod (with the file creation date) to all posts that don't have it."""

import os, sys
import datetime

assert sys.platform == "win32", "This script is only for Windows."
import win32file

POST_FOLDER = os.path.join(os.path.dirname(__file__), "../content/posts")
for a, b, c in os.walk(POST_FOLDER):
    for file in c:
        path = os.path.join(a, file)
        lines = open(path, "r", encoding="utf-8").readlines()
        if any((x.startswith("lastmod:") for x in lines)):
            print(f"[*] {file} already has lastmod. updating.")
        else:
            lines.insert(2)
        open_file_handle = lambda path: win32file.CreateFile(
            path,
            0x101,  # FILE_READ_DATA | FILE_WRITE_ATTRIBUTES
            win32file.FILE_SHARE_READ,
            None,
            win32file.OPEN_EXISTING,
            win32file.FILE_ATTRIBUTE_NORMAL,
            None,
        )
        close_hdl = lambda hdl: win32file.CloseHandle(hdl)
        # modifying `lastmod` changes these too. use win32 apis to revert them later on
        handle = open_file_handle(path)
        create, access, modify = win32file.GetFileTime(handle)
        close_hdl(handle)

        stat = os.stat(path)
        lastmod = modify.astimezone().isoformat()
        print(f"[-] {file} : {lastmod}")
        lines[2] = f"lastmod: {lastmod}\n"
        open(path, "w", encoding="utf-8").writelines(lines)

        handle = open_file_handle(path)
        win32file.SetFileTime(handle, create, access, modify)
        close_hdl(handle)
