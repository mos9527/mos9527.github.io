import os

POST_FOLDER = os.path.join(os.path.dirname(__file__), "../content/posts/cp")
POST_ORDER = ["cp", "segment-tree", "bst", "gcd", "hashing", "dp"]
POST_LANG = "cn"
OUTPUT_FILE = "generated-cp-compilation.md"

with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
    posts = os.listdir(POST_FOLDER)
    posts = [
        next(post for post in posts if kw in post and POST_LANG in post)
        for kw in POST_ORDER
    ]
    for post in posts:
        print(post)
        post_path = os.path.join(POST_FOLDER, post)
        with open(post_path, "r", encoding="utf-8") as input_file:
            sta = 0
            while line := input_file.readline():
                if sta == 0 and line.startswith("---"):
                    sta = 1
                elif sta == 1 and line.startswith("---"):
                    sta = 2
                elif sta == 1 and line.startswith("title: "):
                    title = line.split("title: ")[1].strip()
                    title = title.split("-")[-1].strip()
                    output_file.write(f"# {title}\n\n")
                elif sta == 2:
                    output_file.write(line)
            output_file.write("\n")
