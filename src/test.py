import re
words = re.findall("[a-zA-Z'-]+", "how are you")

sent = ' '.join(w for w in words)
sent