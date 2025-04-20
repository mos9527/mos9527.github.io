import re, os, requests, logging, coloredlogs
from concurrent.futures import ThreadPoolExecutor

coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

PATTERN = re.compile(
    r"https:\/\/github.com\/user-attachments\/assets\/[0-9a-f]*-[0-9a-f]*-[0-9a-f]*-[0-9a-f]*-[0-9a-f]*"
)
FNAME_PATTERN = re.compile(
    r"(?<=\/)[0-9a-f]*-[0-9a-f]*-[0-9a-f]*-[0-9a-f]*-[0-9a-f]*-[0-9a-f]*.*(?=\?)"
)
DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POSTS_DIR = os.path.join(DIRECTORY, "content", "posts")
ASSETS_DIR = os.path.join(DIRECTORY, "static", "image-github")
dl_executor = ThreadPoolExecutor(max_workers=4)


def dl_job(url, root):
    try:
        response = requests.get(url.strip(), stream=True)
        if response.status_code == 200:
            fname = FNAME_PATTERN.findall(response.url)[0]
            path = os.path.join(ASSETS_DIR, fname)
            with open(path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"{url} -> {fname} ok.")
            return path
        else:
            logger.error(f"{url} failed with status code {response.status_code}")
    except Exception as e:
        logger.error(f"{url} failed: {e!r}")
    return None


def thread_job(fpath, root, matches):
    jobs = {match: dl_executor.submit(dl_job, match, root) for match in matches}
    done = dict()
    while not len(jobs) == len(done):
        for match, job in jobs.items():
            if not match in done:
                if job.done():
                    done[match] = job.result()

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    for match, fname in done.items():
        if fname:
            content = content.replace(
                match, os.path.join("static", "image-github", fname)
            )
            logger.info(f"Replaced {match} with {fname} in {fpath}")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)


for root, dirs, files in os.walk(POSTS_DIR):
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            matches = PATTERN.findall(content)
            thread_job(path, root, matches)
