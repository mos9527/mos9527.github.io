import os

POST_FOLDER = os.path.join(os.path.dirname(__file__), "../content/posts/cp")
POST_ORDER = ["cp", "segment-tree", "bst", "gcd", "hashing", "fft", "dp"]
OUTPUT_FILE = "generated-cp-compilation.md"

with open(OUTPUT_FILE, "w") as output_file:
    posts = os.listdir(POST_FOLDER)
    posts = [next(post for post in posts if kw in post) for kw in POST_ORDER]
    for post in posts:
        post_path = os.path.join(POST_FOLDER, post)
        with open(post_path, "r") as input_file:
            content = input_file.read()
            output_file.write(content)
            output_file.write("\n\n")
