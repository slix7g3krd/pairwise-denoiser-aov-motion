# pairwise-denoiser-aov-motion

## 领域

游戏、图形与多媒体

## 主题

降噪 AOV 与运动矢量

## 说明

通道对齐、透明层速度与历史有效性。

初始状态下 `graphics/denoiser_aov_motion.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
