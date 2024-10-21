import os

ROOT_PATH = "/root/autodl-fs/vista_demos"
os.makedirs(ROOT_PATH,exist_ok=True)

actions = ["right","left","stop","forward"]

for i in range(1,11):
    for action in actions:
        url = f"https://vista-demo.github.io/assets/action_control/{i}-{action}.mp4"
        path = f"{ROOT_PATH}/{action}-{i}.mp4"
        cmd = f"wget -O {path} {url}"
        print(cmd)
        os.system(cmd)