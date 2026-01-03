# AI 图像鉴别工具（云端部署版）
专攻 Flux/Midjourney v6/SDXL 高写实AI图片检测的云端网页服务

## 🚀 快速部署（推荐Hugging Face Spaces）
### 步骤1：Fork本仓库
点击GitHub页面右上角的Fork按钮，将仓库复制到你的账号

### 步骤2：部署到Hugging Face Spaces
1. 打开 [Hugging Face Spaces](https://huggingface.co/spaces)
2. 点击 "Create new Space"
3. 填写Space名称，选择 "Gradio" 作为SDK
4. 选择 "Public"（免费），点击 "Create Space"
5. 在Space页面点击 "Files" → "Add file" → "Import from GitHub"
6. 粘贴你的GitHub仓库地址，导入所有文件
7. 等待自动安装依赖并启动（首次启动需下载模型，约5-10分钟）

### 其他部署方式
- **Render**: 新建Web Service，导入GitHub仓库，设置启动命令 `python app.py`，环境变量 `PYTHON_VERSION=3.10`
- **Railway**: 新建Project，导入GitHub仓库，自动识别Gradio，无需额外配置
- **Google Colab**: 直接运行app.py，使用 `demo.launch(share=True)` 获取临时链接

## 📱 使用说明
1. 打开部署后的网页
2. 拖入或上传需要检测的图片
3. 点击"开始深度扫描"
4. 查看鉴定报告（包含AI概率、特征分析）

## ⚡ 性能说明
- 首次访问：需下载模型（约1GB），等待1-2分钟
- 后续访问：CPU环境下单张图片检测约3-5秒
- 支持图片格式：JPG/PNG/WebP，最大分辨率4096x4096

## 📌 注意事项
- 免费部署平台有资源限制（CPU/内存/流量）
- 模型仅用于检测AI生成图片，请勿用于商业用途
- 高分辨率图片检测时间会稍长，建议压缩至2048px以内
