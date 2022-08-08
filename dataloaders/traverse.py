import os
def walkFile(file):
    for root, dirs, files in os.walk(file):
        count = 0
        for f in files:
            count += 1
        if count < 17 and count != 0:
            print(root,"文件数量一共为:",count)

if __name__ == '__main__':
    walkFile(r"/student1/hl_lei/InfantMulti/video2pic2")