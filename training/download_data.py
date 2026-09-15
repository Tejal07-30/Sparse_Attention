from urllib.request import urlretrieve
from pathlib import Path

Path("training/data").mkdir(exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

urlretrieve(url, "training/data/input.txt")

print("downloaded!")