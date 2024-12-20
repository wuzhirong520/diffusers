import numpy as np
cond = {
    "traj": [
        2.2737367544323206e-13,
        -2.2737367544323206e-13,
        -0.2185958982761349,
        3.0310843441404813,
        -0.7780475697172733,
        5.508409468349328,
        -1.7507286850486707,
        7.839336037129783,
        -3.1473881853735293,
        9.892510994069426
    ],
    "speed": [
        17.57,
        18.27,
        18.27,
        18.16,
        17.86
    ],
    "angle": [
        47.70000000000027,
        133.70000000000027,
        158.9000000000001,
        181.9000000000001,
        212.5999999999999
    ],
}

traj = np.array(cond["traj"]).reshape(-1,2)
print(traj)

# a = np.sqrt(traj[0,0]**2+traj[0,1]**2)
# b = np.sqrt((traj[1,0]-traj[0,0])**2+(traj[1,1]-traj[0,1])**2)
# c = np.sqrt((traj[2,0]-traj[1,0])**2+(traj[2,1]-traj[1,1])**2)
# d = np.sqrt((traj[3,0]-traj[2,0])**2+(traj[3,1]-traj[2,1])**2)
# e = np.sqrt((traj[4,0]-traj[3,0])**2+(traj[4,1]-traj[3,1])**2)

# print(a,b,c,d,e)

import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 1. 给定五个二维点
points = traj

# 2. 拟合 B 样条曲线
# splprep 返回一个包含样条拟合的信息的元组，其中第一个元素是参数，第二个元素是 B 样条基函数的次数
tck, u = splprep(points.T, s=0)

# 3. 生成更细分的点用于绘制平滑曲线
# 在 [0, 1] 范围内创建 100 个点来计算曲线
new_u = np.linspace(0, 1, 100)
new_points = np.array(splev(new_u, tck))

# 4. 可视化
plt.figure(figsize=(8, 6))

# 绘制原始点
plt.scatter(points[:, 0], points[:, 1], color='red', label='Original Points')

# 绘制拟合的 B 样条曲线
plt.plot(new_points[0], new_points[1], label='B-spline Curve', color='blue')

# 设置标签
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D B-spline Curve Fit')

# 显示图例
plt.legend()

# 展示图形
plt.grid(True)
plt.show()