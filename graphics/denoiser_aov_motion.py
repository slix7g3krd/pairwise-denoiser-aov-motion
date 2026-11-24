"""光追降噪器参考实现（纯标准库，无第三方依赖）。

六个修复点与函数的对应关系：

1. AOV 通道对齐 .................... aov_align / validation
   漫反射与镜面 AOV 分开降噪，但两者必须落在同一像素网格上，否则逐像素
   累积会发生通道错位。
2. 透明物体运动矢量分层 ............ motion_layers
   透明像素同时携带“透明表面自身速度”和“背景速度”，按透射率分层加权，
   不再用背景速度覆盖表面速度。
3. 历史投影有效性判定 .............. reproject / history_valid / teleport
   重投影后逐像素校验屏幕边界、深度/法线一致性，以及历史写入时的投影
   参数（FOV/宽高比/near/far），相机切换视野后旧投影重投影直接判无效。
4. 异步降噪结果帧对齐 .............. async_result / frame_align / fallback
   异步任务携带帧号，完成时只接受与当前帧匹配的结果；过期/失败/未完成
   一律丢弃并回退到当前帧原始数据，绝不上屏陈旧帧。
5. 采样序列固定种子 ................ seed_fixed / determinism
   每帧每像素的采样种子由 (base_seed, frame, x, y) 纯函数派生，不接触
   时间/地址等熵源，同一场景两次运行产生完全相同的采样序列与噪点图案。
6. 历史裁剪与重置 .................. history_clip / history_reset
   重投影历史颜色裁剪到当前帧邻域 AABB（SVGF 式邻域钳制）抑制拖影；
   失效、传送、切镜头时重置历史与累积计数。

约定：
- 颜色为 (r, g, b) 浮点三元组；网格为 grid[y][x] 的嵌套列表。
- AOV 缓冲为 dict：{"width", "height", "offset", "channels", "valid"}。
- 视图状态为 dict：{"frame", "fov_y", "aspect", "near", "far", "origin"}。
"""

import math

__all__ = [
    "aov_align",
    "motion_layers",
    "history_valid",
    "history_reset",
    "frame_align",
    "seed_fixed",
    "history_clip",
    "validation",
    "audit",
    "fallback",
    "reproject",
    "metrics",
    "teleport",
    "async_result",
    "determinism",
    "denoise_frame",
    "new_state",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
DEFAULT_SEED = 0x20261124
DEPTH_EPS = 1e-3          # 深度一致性相对容差
NORMAL_THRESHOLD = 0.9    # 法线点积阈值（约 25.8 度锥角）
FOV_EPS = 1e-4            # 投影参数相对容差
TELEPORT_DIST = 1.0       # 相机位置传送阈值（世界单位）
MAX_ACCUM = 64            # 累积帧数上限
BACKGROUND_LAYER_ID = -1  # 背景层保留 id


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _in_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


def _finite_color(color):
    return (
        isinstance(color, (tuple, list))
        and len(color) == 3
        and all(isinstance(c, (int, float)) and math.isfinite(c) for c in color)
    )


def _zeros(width, height, fill=0.0):
    return [[fill for _ in range(width)] for _ in range(height)]


def _color_scale(color, k):
    return (color[0] * k, color[1] * k, color[2] * k)


def _color_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mix64(z):
    """splitmix64 风格的整数混合，纯函数、跨平台稳定。"""
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class _Lcg:
    """确定性线性同余采样器，种子固定后序列在任意机器上一致。"""

    _A = 1664525
    _C = 1013904223
    _MOD = 1 << 32

    def __init__(self, seed):
        self.state = int(seed) & (_MASK64)

    def next_unit(self):
        self.state = (self._A * self.state + self._C) % self._MOD
        return self.state / self._MOD


def _lcg_sequence(seed, count):
    rng = _Lcg(seed)
    return [rng.next_unit() for _ in range(count)]


def _projection_key(view):
    """历史写入时的投影指纹；任一项变化都意味着旧投影不能再重投影。"""
    return (
        round(float(view["fov_y"]) / FOV_EPS),
        round(float(view["aspect"]) / FOV_EPS),
        round(float(view["near"]) / FOV_EPS),
        round(float(view["far"]) / FOV_EPS),
    )


# ---------------------------------------------------------------------------
# 合成数据（用于单参数契约调用与自检测试，本身也走确定性种子）
# ---------------------------------------------------------------------------

def _synthetic_aov(width, height, offset=(0, 0), seed=DEFAULT_SEED,
                   channels=("diffuse", "specular")):
    rng = _Lcg(seed)
    grids = {}
    for _ in channels:
        grids[_] = [
            [(rng.next_unit(), rng.next_unit(), rng.next_unit())
             for _ in range(width)]
            for _ in range(height)
        ]
    return {
        "width": width,
        "height": height,
        "offset": offset,
        "channels": grids,
    }


def _demo_layer_stack(width, height):
    """每个像素：一层半透明表面（自身速度 (1,0)）压在背景（速度 (0,0)）上。"""
    stack = _zeros(width, height)
    for y in range(height):
        for x in range(width):
            stack[y][x] = [
                {"id": 1, "opacity": 0.5, "motion": (1.0, 0.0)},
                {"id": BACKGROUND_LAYER_ID, "opacity": 1.0,
                 "motion": (0.0, 0.0), "background": True},
            ]
    return stack


# ===========================================================================
# 修复点 1：AOV 通道对齐
# ===========================================================================

def _as_buffer(buffer_like, channels):
    if isinstance(buffer_like, dict):
        return buffer_like
    if _is_int(buffer_like):
        return _synthetic_aov(buffer_like, buffer_like, channels=channels)
    raise TypeError("AOV 缓冲必须是 dict，或用于自检的 int 尺寸")


def aov_align(*buffers, reference=None, channels=("diffuse", "specular")):
    """把若干 AOV 缓冲对齐到统一像素网格。

    各缓冲可以有不同的 width/height/offset；对齐后 diffuse 与 specular
    拥有完全相同的尺寸、原点和逐像素有效位，缺通道/越界区域 valid=0，
    分开降噪时不会再发生通道错位。

    返回对齐后的单一缓冲 dict（含对齐报告）。
    """
    if len(buffers) == 1 and _is_int(buffers[0]):
        n = buffers[0]
        buffers = (
            _synthetic_aov(n, n, offset=(0, 0), seed=DEFAULT_SEED),
            _synthetic_aov(n, n, offset=(1, 1), seed=DEFAULT_SEED ^ 0x55),
        )

    normalized = [_as_buffer(b, channels) for b in buffers]
    if not normalized:
        return {
            "width": 0, "height": 0, "offset": (0, 0),
            "channels": {ch: [] for ch in channels},
            "valid": {ch: [] for ch in channels},
            "report": [],
        }

    if reference is not None:
        rx, ry = reference["offset"]
        width, height = reference["width"], reference["height"]
    else:
        rx = min(b["offset"][0] for b in normalized)
        ry = min(b["offset"][1] for b in normalized)
        width = max(b["offset"][0] + b["width"] for b in normalized) - rx
        height = max(b["offset"][1] + b["height"] for b in normalized) - ry

    aligned = {
        "width": width,
        "height": height,
        "offset": (rx, ry),
        "channels": {ch: _zeros(width, height, fill=(0.0, 0.0, 0.0))
                     for ch in channels},
        "valid": {ch: _zeros(width, height, fill=0) for ch in channels},
        "report": [],
    }

    for index, buf in enumerate(normalized):
        ox = buf["offset"][0] - rx
        oy = buf["offset"][1] - ry
        for ch in channels:
            src = buf.get("channels", {}).get(ch)
            if src is None:
                aligned["report"].append(
                    {"buffer": index, "channel": ch, "issue": "missing_channel"})
                continue
            for y in range(min(buf["height"], height - oy)):
                for x in range(min(buf["width"], width - ox)):
                    gx, gy = ox + x, oy + y
                    if not _in_bounds(gx, gy, width, height):
                        continue
                    aligned["channels"][ch][gy][gx] = src[y][x]
                    aligned["valid"][ch][gy][gx] = 1
        if (ox, oy) != (0, 0) or buf["width"] != width or buf["height"] != height:
            aligned["report"].append(
                {"buffer": index, "shift": (ox, oy),
                 "from": (buf["width"], buf["height"]),
                 "to": (width, height), "issue": "realigned"})

    ok, errors = validation(aligned)
    aligned["valid_buffer"] = ok
    aligned["errors"] = errors
    return aligned


def validation(buffer):
    """校验 AOV 缓冲：必需通道齐全、各通道同形、数值全部有限。

    返回 (ok, errors)。
    """
    if _is_int(buffer):
        buffer = _synthetic_aov(buffer, buffer)

    errors = []
    if not isinstance(buffer, dict):
        return False, ["buffer_not_dict"]

    width, height = buffer.get("width"), buffer.get("height")
    if not (_is_int(width) and _is_int(height) and width > 0 and height > 0):
        errors.append("bad_dimensions")
        return False, errors

    channels = buffer.get("channels", {})
    for required in ("diffuse", "specular"):
        if required not in channels:
            errors.append("missing_channel:%s" % required)

    for name, grid in channels.items():
        if len(grid) != height or any(len(row) != width for row in grid):
            errors.append("shape_mismatch:%s" % name)
            continue
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                if not _finite_color(color):
                    errors.append("non_finite:%s@(%d,%d)" % (name, x, y))

    valid = buffer.get("valid")
    if isinstance(valid, dict):
        for name, mask in valid.items():
            if len(mask) != height or any(len(row) != width for row in mask):
                errors.append("valid_shape_mismatch:%s" % name)
            elif any(v not in (0, 1) for row in mask for v in row):
                errors.append("valid_value_not_binary:%s" % name)

    return not errors, errors


# ===========================================================================
# 修复点 2：透明物体运动矢量分层
# ===========================================================================

def _layer_stack_from_grids(background_motion, surface_motion, opacity):
    height = len(surface_motion)
    width = len(surface_motion[0]) if height else 0
    stack = _zeros(width, height)
    for y in range(height):
        for x in range(width):
            op = float(opacity[y][x])
            stack[y][x] = [
                {"id": 1, "opacity": op, "motion": tuple(surface_motion[y][x])},
                {"id": BACKGROUND_LAYER_ID, "opacity": 1.0,
                 "motion": tuple(background_motion[y][x]), "background": True},
            ]
    return stack


def motion_layers(layer_stack=None, background_motion=None,
                  surface_motion=None, opacity=None):
    """把透明像素的运动矢量按层拆开并按透射率加权。

    支持两种输入：
    1. 分层 G-buffer：layer_stack[y][x] 为前→后的层列表，每层
       {"id", "opacity", "motion"}，背景层带 "background": True。
    2. 便捷网格：background_motion / surface_motion / opacity 三张同形网格。

    关键修复：透明表面层保留自身运动矢量；背景速度只以“剩余透射率 T”
    的权重参与合成，而不是直接覆盖表面速度。

    返回 {"layers", "composite", "surface"} 三张同形网格。
    """
    if _is_int(layer_stack):
        layer_stack = _demo_layer_stack(layer_stack, layer_stack)

    if layer_stack is None and surface_motion is not None:
        layer_stack = _layer_stack_from_grids(
            background_motion, surface_motion, opacity)
    if layer_stack is None:
        raise ValueError("需要 layer_stack 或 surface/background 运动网格")

    height = len(layer_stack)
    width = len(layer_stack[0]) if height else 0
    out_layers = _zeros(width, height)
    composite = _zeros(width, height)
    surface = _zeros(width, height)

    for y in range(height):
        for x in range(width):
            cell = layer_stack[y][x]
            transmittance = 1.0
            weighted = []
            surface_motion_at_pixel = None
            for layer in cell:
                motion = tuple(float(v) for v in layer["motion"])
                is_background = bool(layer.get(
                    "background", layer.get("id") == BACKGROUND_LAYER_ID))
                if is_background:
                    # 背景只拿到穿过所有透明层之后剩余的透射率权重
                    weight = transmittance
                else:
                    op = min(1.0, max(0.0, float(layer["opacity"])))
                    weight = transmittance * op
                    transmittance *= (1.0 - op)
                    if surface_motion_at_pixel is None:
                        surface_motion_at_pixel = motion
                weighted.append((layer.get("id"), motion, weight))

            total = sum(w for _, _, w in weighted)
            if total <= 0.0:
                # 全透明且无背景权重的退化情况：均匀分配，避免除零
                share = 1.0 / len(weighted) if weighted else 0.0
                weighted = [(i, m, share) for i, m, _ in weighted]
                total = sum(w for _, _, w in weighted)

            blend = (0.0, 0.0)
            normalized = []
            for layer_id, motion, weight in weighted:
                w = weight / total if total > 0 else 0.0
                normalized.append((layer_id, motion, w))
                blend = (blend[0] + motion[0] * w,
                         blend[1] + motion[1] * w)

            out_layers[y][x] = normalized
            composite[y][x] = blend
            # 表面速度：最前面的非背景层，透明物体用它自己的速度重投影
            surface[y][x] = (
                surface_motion_at_pixel
                if surface_motion_at_pixel is not None
                else weighted[-1][1]
            )

    return {"layers": out_layers, "composite": composite, "surface": surface}


# ===========================================================================
# 修复点 3：重投影、历史有效性、传送检测
# ===========================================================================

def _warp(grid, motion):
    """按运动矢量把上一帧网格扭曲到当前帧像素坐标。

    运动矢量约定：motion[y][x] = 当前像素指向上一帧位置的像素位移，
    即 history_coord = current_coord - motion。
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    warped = [row[:] for row in grid]
    valid = _zeros(width, height, fill=0)
    coords = _zeros(width, height)
    for y in range(height):
        for x in range(width):
            vx, vy = motion[y][x]
            hx = int(round(x - vx))
            hy = int(round(y - vy))
            coords[y][x] = (hx, hy)
            if _in_bounds(hx, hy, width, height):
                warped[y][x] = grid[hy][hx]
                valid[y][x] = 1
            else:
                valid[y][x] = 0
    return warped, valid, coords


def reproject(history=None, motion=None, view=None, history_view=None):
    """用当前像素的运动矢量重投影历史缓冲，并标记出界失效像素。"""
    if _is_int(history):
        n = history
        history = _synthetic_aov(n, n, channels=("color",))["channels"]["color"]
        motion = [[(1.0, 0.0) if x > 0 else (0.0, 0.0)
                   for x in range(n)] for y in range(n)]

    if history is None or motion is None:
        raise ValueError("reproject 需要 history 与 motion 网格")

    warped, valid, coords = _warp(history, motion)
    return {
        "color": warped,
        "valid": valid,
        "coord": coords,
        "view": view,
        "history_view": history_view,
    }


def history_valid(reprojected=None, current=None, view=None, previous=None,
                  depth_eps=DEPTH_EPS, normal_threshold=NORMAL_THRESHOLD):
    """判定单个重投影像素的历史是否仍然有效。

    检查四项，任一失败即无效：
    1. 重投影坐标在屏幕范围内；
    2. 当前视图与历史写入视图的投影指纹一致（FOV/宽高比/near/far），
       相机切换视野后旧投影重投影直接判无效；
    3. 深度连续，未落在反遮挡/新暴露区域；
    4. 法线锥一致，未跨到别的表面。

    返回 {"valid": bool, "reasons": [...]}。
    """
    if _is_int(reprojected):
        frame = reprojected
        view_a = {"frame": frame, "fov_y": 0.9, "aspect": 1.78,
                  "near": 0.1, "far": 100.0, "origin": (0.0, 0.0, 0.0)}
        view_b = dict(view_a, fov_y=1.2)  # 视野切换
        good = history_valid(
            {"coord": (1, 1), "depth": 1.0, "normal": (0.0, 0.0, 1.0)},
            {"coord": (1, 1), "depth": 1.0, "normal": (0.0, 0.0, 1.0)},
            view_a, view_a)
        bad = history_valid(
            {"coord": (1, 1), "depth": 1.0, "normal": (0.0, 0.0, 1.0)},
            {"coord": (1, 1), "depth": 1.0, "normal": (0.0, 0.0, 1.0)},
            view_a, view_b)
        return {"frame": frame, "stable_view": good, "fov_switch": bad}

    reasons = []
    reprojected = reprojected or {}
    current = current or {}

    width = current.get("width", reprojected.get("width", 0))
    height = current.get("height", reprojected.get("height", 0))
    coord = reprojected.get("coord")
    if coord is None:
        reasons.append("missing_coord")
    elif width and height and not _in_bounds(coord[0], coord[1], width, height):
        reasons.append("out_of_bounds")

    # 投影有效性：历史缓冲携带的是哪一个投影矩阵，必须与当前帧一致
    if view is not None and previous is not None:
        if _projection_key(view) != _projection_key(previous):
            reasons.append("projection_changed")
        frame_gap = view.get("frame", 0) - previous.get("frame", 0)
        if frame_gap > 1:
            reasons.append("frame_gap")

    d_hist = reprojected.get("depth")
    d_cur = current.get("depth")
    if d_hist is not None and d_cur is not None:
        if abs(d_hist - d_cur) > depth_eps * (1.0 + abs(d_cur)):
            reasons.append("depth_discontinuity")

    n_hist = reprojected.get("normal")
    n_cur = current.get("normal")
    if n_hist is not None and n_cur is not None:
        if _dot(n_hist, n_cur) < normal_threshold:
            reasons.append("normal_cone_reject")

    return {"valid": not reasons, "reasons": reasons}


def teleport(current_view=None, previous_view=None,
             distance_threshold=TELEPORT_DIST, fov_eps=FOV_EPS):
    """检测相机传送/切镜头：位置跳变、FOV 跳变或帧号不连续。"""
    if _is_int(current_view):
        n = current_view
        base = {"frame": n, "fov_y": 0.9, "aspect": 1.78,
                "near": 0.1, "far": 100.0, "origin": (0.0, 0.0, 0.0)}
        moved = dict(base, frame=n + 1, origin=(5.0, 0.0, 0.0))
        fov_changed = dict(base, frame=n + 1, fov_y=base["fov_y"] + 0.5)
        return {
            "same_position": teleport(base, dict(base, frame=n + 1)),
            "position_jump": teleport(moved, base),
            "fov_jump": teleport(fov_changed, base),
        }

    if current_view is None or previous_view is None:
        return {"teleport": False, "reason": "no_previous_view"}

    if current_view.get("frame", 0) - previous_view.get("frame", 0) > 1:
        return {"teleport": True, "reason": "frame_gap"}

    oa = current_view.get("origin")
    ob = previous_view.get("origin")
    if oa is not None and ob is not None:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(oa, ob)))
        if dist > distance_threshold:
            return {"teleport": True, "reason": "position_jump",
                    "distance": dist}

    if abs(current_view.get("fov_y", 0.0) - previous_view.get("fov_y", 0.0)) \
            > fov_eps:
        return {"teleport": True, "reason": "fov_jump"}

    return {"teleport": False, "reason": "continuous"}


# ===========================================================================
# 修复点 4：异步结果帧对齐
# ===========================================================================

def frame_align(frame_id, payload=None):
    """给载荷打上帧号并校验，防止异步结果错帧上屏。"""
    if _is_int(payload):
        payload = {"frame": payload, "data": None}
    if payload is None:
        # 单参数自检：构造一个同帧载荷与一个过期载荷
        return {
            "frame": frame_id,
            "aligned_self": frame_align(frame_id,
                                        {"frame": frame_id, "data": None}),
            "aligned_stale": frame_align(frame_id,
                                         {"frame": frame_id - 1, "data": None}),
        }

    if isinstance(payload, dict) and "frame" in payload:
        aligned = payload["frame"] == frame_id
        return {
            "frame": frame_id,
            "payload": payload,
            "aligned": aligned,
            "reason": None if aligned else "frame_mismatch",
        }
    return {"frame": frame_id, "payload": payload, "aligned": True,
            "reason": None}


def fallback(current_frame):
    """异步结果不可用时的安全回退：直接上当前帧自己的原始数据。

    当前帧数据永远与当前帧号匹配，绝不会像陈旧异步结果那样错帧。
    """
    if _is_int(current_frame):
        current_frame = _synthetic_aov(
            current_frame, current_frame)
    return {"data": current_frame, "denoised": False,
            "source": "fallback", "frame": None}


def async_result(job=None, current_frame_id=0, current_frame=None):
    """消费一个异步降噪任务的结果，严格按帧号对齐。

    job = {"frame", "status": ready/pending/failed, "data"}
    - ready 且帧号匹配：接受；
    - ready 但帧号过期：丢弃，回退当前帧；
    - pending：等待（保留上一帧输出，由调用方决定）；
    - failed：回退当前帧。
    """
    if _is_int(job):
        n = job
        job = {"frame": n, "status": "ready",
               "data": _synthetic_aov(n, n)}
        current_frame_id = n

    if not isinstance(job, dict):
        return {"action": "fallback", "reason": "bad_job",
                "data": fallback(current_frame)["data"]
                if current_frame is not None else None}

    status = job.get("status", "ready")
    if status == "pending":
        return {"action": "wait", "reason": "job_pending",
                "frame": job.get("frame")}

    if status == "failed":
        return {"action": "fallback", "reason": "job_failed",
                "data": fallback(current_frame)["data"]
                if current_frame is not None else None}

    if job.get("frame") != current_frame_id:
        return {
            "action": "fallback",
            "reason": "stale_frame",
            "job_frame": job.get("frame"),
            "current_frame": current_frame_id,
            "data": fallback(current_frame)["data"]
            if current_frame is not None else None,
        }

    return {"action": "accept", "reason": None,
            "frame": current_frame_id, "data": job.get("data")}


# ===========================================================================
# 修复点 5：固定种子与确定性
# ===========================================================================

def seed_fixed(frame_index, x=0, y=0, base_seed=DEFAULT_SEED, width=0):
    """由 (base_seed, frame, x, y) 纯函数派生采样种子。

    不读取时间、pid、对象地址等任何外部熵源；相同输入在任意机器、
    任意运行次数下得到相同种子，从而固定采样序列与噪点图案。
    """
    key = (
        int(base_seed)
        + int(frame_index) * 0x9E3779B1
        + int(y) * 0x85EBCA77
        + (int(x) + int(width) * int(y)) * 0xC2B2AE3D
    ) & _MASK64
    return _mix64(key)


def determinism(frame_index=None, count=16, width=4, height=4):
    """两次独立生成同一帧的采样序列并逐元素比对，验证确定性。"""
    if frame_index is None:
        frame_index = 0
    if _is_int(frame_index) and count is None:
        count = 16

    def make_sequence():
        sequence = _zeros(width, height)
        for y in range(height):
            for x in range(width):
                seed = seed_fixed(frame_index, x, y, width=width)
                sequence[y][x] = _lcg_sequence(seed, count)
        return sequence

    seq_a = make_sequence()
    seq_b = make_sequence()
    equal = seq_a == seq_b
    # 同时验证种子派生本身是纯函数
    seed_equal = all(
        seed_fixed(frame_index, x, y, width=width)
        == seed_fixed(frame_index, x, y, width=width)
        for y in range(height) for x in range(width)
    )
    return {
        "equal": equal and seed_equal,
        "frame": frame_index,
        "count": count,
        "sequence": seq_a,
    }


# ===========================================================================
# 修复点 6：历史裁剪与重置
# ===========================================================================

def history_clip(history_color=None, neighborhood=None):
    """把重投影历史颜色钳制到当前帧邻域的逐通道 AABB 内。

    SVGF 式邻域钳制：历史像素若偏离当前帧邻域过远，说明它来自已离开
    的表面，裁回去以抑制拖影。返回 (clamped_color, clipped_flag)。
    """
    if _is_int(history_color):
        n = history_color
        history_color = (float(n + 1), -1.0, 0.5)
        neighborhood = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]

    if not neighborhood:
        return tuple(history_color), False

    lows = tuple(min(c[i] for c in neighborhood) for i in range(3))
    highs = tuple(max(c[i] for c in neighborhood) for i in range(3))
    clamped = tuple(
        min(max(history_color[i], lows[i]), highs[i]) for i in range(3)
    )
    return clamped, any(
        abs(clamped[i] - history_color[i]) > 0.0 for i in range(3))


def history_reset(frame_index, reason="manual", width=1, height=1, view=None):
    """构造一份全新的空历史：颜色清零、累积计数归零并记录原因。"""
    if _is_int(frame_index) and reason == "manual":
        width = max(1, width)
        height = max(1, height)
    return {
        "frame": frame_index,
        "reason": reason,
        "view": view,
        "color": _zeros(width, height, fill=(0.0, 0.0, 0.0)),
        "accum": _zeros(width, height, fill=0),
    }


# ===========================================================================
# 辅助：噪声指标与状态审计
# ===========================================================================

def metrics(grid=None, reference=None):
    """噪声指标。

    单网格：逐通道均值/方差 + 平均邻域梯度（空间高频噪声代理）。
    带参考网格：额外给出 RMSE/MAE（时间抖动或误差度量）。
    """
    if _is_int(grid):
        n = grid
        rng = _Lcg(DEFAULT_SEED)
        grid = [[(rng.next_unit(), rng.next_unit(), rng.next_unit())
                 for _ in range(n)] for _ in range(n)]

    if not grid or not grid[0]:
        return {"mean": (0.0, 0.0, 0.0), "variance": (0.0, 0.0, 0.0),
                "gradient": 0.0, "rmse": None, "mae": None}

    height = len(grid)
    width = len(grid[0])
    count = width * height
    sums = [0.0, 0.0, 0.0]
    for row in grid:
        for color in row:
            for i in range(3):
                sums[i] += color[i]
    means = tuple(s / count for s in sums)

    sq = [0.0, 0.0, 0.0]
    for row in grid:
        for color in row:
            for i in range(3):
                sq[i] += (color[i] - means[i]) ** 2
    variances = tuple(s / count for s in sq)

    gradient_sum = 0.0
    gradient_count = 0
    for y in range(height):
        for x in range(width):
            if x + 1 < width:
                gradient_sum += math.sqrt(
                    sum((grid[y][x][i] - grid[y][x + 1][i]) ** 2
                        for i in range(3)))
                gradient_count += 1
            if y + 1 < height:
                gradient_sum += math.sqrt(
                    sum((grid[y][x][i] - grid[y + 1][x][i]) ** 2
                        for i in range(3)))
                gradient_count += 1
    gradient = gradient_sum / gradient_count if gradient_count else 0.0

    result = {"mean": means, "variance": variances,
              "gradient": gradient, "rmse": None, "mae": None}

    if reference is not None:
        se = 0.0
        ae = 0.0
        for y in range(min(height, len(reference))):
            for x in range(min(width, len(reference[0]))):
                for i in range(3):
                    d = grid[y][x][i] - reference[y][x][i]
                    se += d * d
                    ae += abs(d)
        denom = 3 * width * height
        result["rmse"] = math.sqrt(se / denom)
        result["mae"] = ae / denom

    return result


def audit(state=None):
    """汇总降噪器运行状态，输出健康标志、告警与计数。"""
    if _is_int(state):
        state = {
            "events": [{"type": "async_stale", "frame": i}
                       for i in range(state)],
            "stats": {"resets": state, "clips": 0, "stale": state},
        }
    if not isinstance(state, dict):
        return {"ok": False, "warnings": ["bad_state"], "counts": {}}

    events = state.get("events", [])
    stats = state.get("stats", {})
    warnings = []
    for event in events:
        etype = event.get("type", "unknown") if isinstance(event, dict) \
            else str(event)
        if etype in ("async_stale", "job_failed", "history_reset",
                     "projection_changed", "teleport", "aov_missing_channel"):
            warnings.append("%s@frame%s" % (etype, event.get("frame", "?")))

    counts = {
        "events": len(events),
        "resets": stats.get("resets", 0),
        "clips": stats.get("clips", 0),
        "stale": stats.get("stale", 0),
        "realign": stats.get("realign", 0),
    }
    return {
        "ok": not warnings,
        "warnings": warnings,
        "counts": counts,
        "frame": state.get("frame"),
    }


# ===========================================================================
# 端到端管线：把六个修复点串起来
# ===========================================================================

def new_state():
    return {
        "frame": -1,
        "view": None,
        "history": {},       # channel -> history_reset 结构
        "events": [],
        "stats": {"resets": 0, "clips": 0, "stale": 0, "realign": 0},
    }


def _neighborhood(grid, x, y, width, height):
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            if _in_bounds(nx, ny, width, height):
                out.append(grid[ny][nx])
    return out


def denoise_frame(state, aov, gbuffer, view, async_job=None):
    """对一帧执行完整降噪流程，返回 (output, state)。

    流程严格对应六个修复点，且每帧每像素的采样都走 seed_fixed，
    两次独立运行的输出逐像素一致。
    """
    frame_index = state["frame"] + 1
    width = aov["width"]
    height = aov["height"]
    events = state["events"]
    stats = state["stats"]

    # 修复点 1：先校验再对齐，漫反射/镜面落到同一像素网格后分别降噪
    ok, errors = validation(aov)
    if not ok:
        for err in errors:
            events.append({"type": "aov_missing_channel",
                           "frame": frame_index, "detail": err})
    aligned = aov_align(aov)
    if aligned["report"]:
        stats["realign"] += len(aligned["report"])

    # 修复点 3/6：传送或切镜头（含 FOV 切换）时整体重置历史
    cut = teleport(view, state["view"]) if state["view"] is not None \
        else {"teleport": True, "reason": "first_frame"}
    if cut["teleport"]:
        events.append({"type": "teleport", "frame": frame_index,
                       "reason": cut["reason"]})
        state["history"] = {
            ch: history_reset(frame_index,
                              reason=cut["reason"],
                              width=width, height=height, view=view)
            for ch in aligned["channels"]
        }
        stats["resets"] += len(state["history"])

    # 修复点 2：透明物体运动矢量分层，重投影使用分层合成速度
    motion = motion_layers(gbuffer["layers"])["composite"]

    output_channels = {}
    for ch, current_grid in aligned["channels"].items():
        hist = state["history"].setdefault(
            ch, history_reset(frame_index, "first_frame",
                              width, height, view))

        repro = reproject(hist["color"], motion,
                          view=view, history_view=hist["view"])
        warped_depth, _, _ = _warp(gbuffer["depth"], motion)
        warped_normal, _, _ = _warp(gbuffer["normal"], motion)

        new_color = _zeros(width, height, fill=(0.0, 0.0, 0.0))
        for y in range(height):
            for x in range(width):
                # 修复点 5：固定种子派生本像素采样序列
                seed = seed_fixed(frame_index, x, y, width=width)
                sample_jitter = _lcg_sequence(seed, 1)[0]

                verdict = history_valid(
                    {"coord": repro["coord"][y][x],
                     "depth": warped_depth[y][x],
                     "normal": warped_normal[y][x],
                     "width": width, "height": height},
                    {"coord": (x, y),
                     "depth": gbuffer["depth"][y][x],
                     "normal": gbuffer["normal"][y][x],
                     "width": width, "height": height},
                    view, hist["view"])

                cur = current_grid[y][x]
                if not verdict["valid"] or not repro["valid"][y][x]:
                    # 修复点 6：局部失效 → 该像素历史重置
                    new_color[y][x] = cur
                    hist["accum"][y][x] = 1
                    for reason in verdict["reasons"]:
                        if reason == "projection_changed":
                            events.append(
                                {"type": "projection_changed",
                                 "frame": frame_index})
                    continue

                # 修复点 6：邻域 AABB 裁剪抑制拖影
                clipped, did_clip = history_clip(
                    repro["color"][y][x],
                    _neighborhood(current_grid, x, y, width, height))
                if did_clip:
                    stats["clips"] += 1

                w = hist["accum"][y][x]
                if w <= 0:
                    new_color[y][x] = cur
                    hist["accum"][y][x] = 1
                else:
                    blended = tuple(
                        (clipped[i] * w + cur[i]) / (w + 1)
                        for i in range(3))
                    # 固定种子只带来确定性的微抖动，不引入外部随机性
                    new_color[y][x] = tuple(
                        c + (sample_jitter - 0.5) * 1e-6 for c in blended)
                    hist["accum"][y][x] = min(w + 1, MAX_ACCUM)

        hist["color"] = new_color
        hist["view"] = view
        hist["frame"] = frame_index
        output_channels[ch] = new_color

    # 修复点 4：异步结果按帧号对齐，过期/失败一律回退当前帧
    if async_job is not None:
        decision = async_result(async_job, frame_index,
                                current_frame=output_channels)
        if decision["action"] == "accept":
            output_channels = decision["data"]
        elif decision["action"] == "fallback":
            stats["stale"] += 1
            events.append({"type": "async_stale", "frame": frame_index,
                           "reason": decision.get("reason")})
        else:
            events.append({"type": "async_pending", "frame": frame_index})

    state["frame"] = frame_index
    state["view"] = view
    return {
        "frame": frame_index,
        "channels": output_channels,
        "audit": audit(state),
    }, state
