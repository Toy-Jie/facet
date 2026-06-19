# 修图开源库取舍

来源参考：PixCake 开源许可页面中的第三方库清单。

## 保留并用于本项目

| 库 / 技术 | 项目落点 | 说明 |
| --- | --- | --- |
| OpenCV | `api/routers/retouch.py` | 核心像素处理：磨皮、祛瑕、inpaint、头发 mask 降级、背景虚化、法令纹局部修复。 |
| Pillow | `api/routers/retouch.py`、缩略图、RAW 预览链路 | 图片读取、裁剪旋转、JPEG 导出、EXIF Orientation 应用、ICC 色彩配置保留。 |
| libjpeg / libpng / libtiff | Pillow 底层能力 | 通过 Pillow 间接使用，负责常见图片格式读写。 |
| lcms2 | Pillow 底层能力 | 通过 Pillow 间接使用，保留 ICC profile，减少导出后色彩偏差。 |
| EXIF / XMP 相关能力 | `exifread`、`exiftool`、Pillow EXIF | 扫描阶段读取元数据；修图保存副本时保留 EXIF，并将 Orientation 归一化为 1。 |
| ONNX Runtime | 可选模型推理后端 | 保留作为本地 AI 模型推理方向，优先用于后续人像分割、修复、增强模型。 |
| rawpy / LibRaw | RAW 显示和转换 | 替代直接接入 DNG SDK，保留当前跨格式 RAW 处理路径。 |
| UniFace / BiSeNet | 头发和脸部区域解析 | 已作为可选依赖接入头发美化；失败时回退 OpenCV。 |

## 暂不直接接入

| 库 / 技术 | 暂不接入原因 |
| --- | --- |
| GPUImage | 更偏移动端 / shader 滤镜栈；当前项目是 Python 后端 + Angular 前端，OpenCV/Pillow 更直接。 |
| MNN | 和 ONNX Runtime 的定位重复；Python 生态和现有模型接入优先选择 ONNX Runtime。 |
| Adobe DNG SDK | 当前 RAW 处理已有 rawpy / LibRaw，维护成本更低。 |
| libyuv | 更偏视频 / YUV 像素格式转换；当前照片修图链路暂不需要。 |
| Eigen / OpenCL / NEON / SSE | 属于底层加速或数学实现，优先通过 OpenCV、ONNX Runtime、Pillow 间接获得。 |
| Filament / GLEW / GLM / OSMesa / Qt OpenGL | 偏 3D/桌面 GUI/渲染，不适合当前 Web 修图界面。 |

## 当前集成策略

修图功能优先保持轻量、本地、可控：

1. 基础和人像修图继续以 OpenCV + Pillow 为主。
2. 模型类能力优先走 ONNX Runtime 或现有 Python 模型生态，避免同时维护多个推理后端。
3. 导出副本不覆盖原图，保留 EXIF / ICC，且把 Orientation 写成 1，避免图片被二次旋转。
4. 只有当某个库能直接改善用户可见效果，才加入依赖；底层库尽量由 Pillow/OpenCV/ONNX Runtime 间接管理。
