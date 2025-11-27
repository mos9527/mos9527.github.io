"""Add lastmod (with the file creation date) to all posts that don't have it."""

import os, sys
import datetime

if sys.platform == "win32":
    import win32file

    class auto_utime:
        @staticmethod
        def __open_file(path):
            return win32file.CreateFile(
                path,
                0x101,  # FILE_READ_DATA | FILE_WRITE_ATTRIBUTES
                win32file.FILE_SHARE_READ,
                None,
                win32file.OPEN_EXISTING,
                win32file.FILE_ATTRIBUTE_NORMAL,
                None,
            )

        @staticmethod
        def __close_file(handle):
            return win32file.CloseHandle(handle)

        def __init__(self, path):
            self.path = path
            handle = self.__open_file(path)
            self.filetime = win32file.GetFileTime(handle)
            self.__close_file(handle)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            handle = self.__open_file(self.path)
            win32file.SetFileTime(handle, *self.filetime)
            self.__close_file(handle)

        @property
        def modify(self):
            return self.filetime[2].astimezone()

else:

    class auto_utime:
        def __init__(self, path):
            self.path = path
            self.stat = os.stat(path)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            os.utime(self.path, (self.stat.st_atime, self.stat.st_mtime))

        @property
        def modify(self):
            return datetime.datetime.fromtimestamp(self.stat.st_mtime)


POST_FOLDER = os.path.join(os.path.dirname(__file__), "../content/posts")
for a, b, c in os.walk(POST_FOLDER):
    for file in c:
        path = os.path.join(a, file)
        try:
            lines = open(path, "r", encoding="utf-8").readlines()
        except UnicodeDecodeError:
            print(f"[!] {file} can't be decoded. skipping.")
            continue
        if any((x.startswith("lastmod:") for x in lines)):
            print(f"[*] {file} already has lastmod. updating.")        

        with auto_utime(path) as ftime:
            lastmod = ftime.modify.isoformat()
            print(f"[-] {file} : {lastmod}")
            lines[2] = f"lastmod: {lastmod}\n"
            open(path, "w", encoding="utf-8").writelines(lines)
