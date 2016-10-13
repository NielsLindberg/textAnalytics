import os

dir = os.path.dirname(__file__)

path = 'D:/Workspace/CBS/BigSocialData/textAnalytics/novel.txt'
print(path)

with open(path, mode="rt", encoding="utf8") as in_file:
    text = in_file.read()

print(text)
