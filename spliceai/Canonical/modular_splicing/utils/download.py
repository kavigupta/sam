import gzip
import tempfile
import requests


def read_gzip(url):
    out = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix=".gz") as temp:
        with open(temp.name, "wb") as f:
            f.write(out.content)
        with gzip.open(temp.name, "rb") as f:
            result = f.read()
    return result
