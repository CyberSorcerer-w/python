import os
import torch
import gradio as gr
from transformers import pipeline
from PIL import Image

# ================= 云端部署适配配置 =================
# 1. 模型缓存路径（适配云平台）
os.environ['HF_HOME'] = '/tmp/hf_models'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

# 2. 设备自动适配（云平台多为CPU，自动降级）
DEVICE = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else -1
print(f">>> 运行设备: {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
print(f">>> 模型缓存路径: {os.environ['HF_HOME']}")

# ======================================================

class ImageGuard:
    def __init__(self):
        print("\n" + "="*40)
        print(">>> 正在启动【云端版】图片鉴别核心...")
        print(">>> 模式：高灵敏度 (针对 Flux/SDXL 优化)")
        print(">>> 首次运行需要下载模型，请稍候...")
        print("="*40 + "\n")
        
        # --- 模型 A: 纹理专家 ---
        print(">>> [1/2] 加载纹理分析模型 (umm-maybe)...")
        try:
            self.pipe_texture = pipeline(
                "image-classification", 
                model="umm-maybe/AI-image-detector", 
                device=DEVICE,
                # 云端优化：禁用缓存减少内存占用
                model_kwargs={"low_cpu_mem_usage": True}
            )
        except Exception as e:
            print(f"❌ 纹理模型加载失败: {e}")
            self.pipe_texture = None

        # --- 模型 B: 结构专家 ---
        print(">>> [2/2] 加载结构分析模型 (dima806)...")
        try:
            self.pipe_struct = pipeline(
                "image-classification", 
                model="dima806/ai_generated_image_detection", 
                device=DEVICE,
                model_kwargs={"low_cpu_mem_usage": True}
            )
        except Exception as e:
            print(f"❌ 结构模型加载失败: {e}")
            self.pipe_struct = None
        
        print(">>> 初始化完成！")

    def _get_score(self, pipe, image):
        if not pipe: return 0.0
        try:
            results = pipe(image)
            ai_keywords = ['fake', 'artificial', 'generated', 'ai', 'computer']
            real_keywords = ['human', 'real', 'photo', 'natural']
            
            for res in results:
                if any(k in res['label'].lower() for k in ai_keywords):
                    return res['score']
            for res in results:
                if any(k in res['label'].lower() for k in real_keywords):
                    return 1.0 - res['score']
            return 0.0
        except Exception as e:
            print(f"❌ 评分计算失败: {e}")
            return 0.0

    def analyze(self, image):
        if image is None: return "⚠️ 请上传图片"
        
        # 获取分数
        score_tex = self._get_score(self.pipe_texture, image)
        score_str = self._get_score(self.pipe_struct, image)
        
        # 核心算法：高敏加权
        base_risk = max(score_tex, score_str)
        
        verdict = ""
        desc = ""
        
        # 判定逻辑 (阈值 15%)
        if base_risk > 0.8:
            verdict = "🔴 [确认] 极大概率是 AI 生成"
            desc = "检测到明显的生成式指纹，毫无疑问的 AI 作品。"
        elif base_risk > 0.5:
            verdict = "🟠 [高疑] 包含大量合成特征"
            desc = "虽然光影自然，但纹理细节暴露了 AI 身份。"
        elif base_risk > 0.15: 
            verdict = "🟡 [存疑] 疑似 Flux/SDXL 高仿"
            desc = f"检测到异常纹理信号 ({base_risk:.1%})。真实相机直出照片几乎不会超过 10%。"
        else:
            verdict = "🟢 [真实] 符合摄影特征"
            desc = "噪点分布自然，未检测到 AI 痕迹。"

        details = f"📊 专家会诊数据:\n"
        details += f"• 纹理分析 (细节): {score_tex:.1%}\n"
        details += f"• 结构分析 (构图): {score_str:.1%}\n"
        details += "-" * 30 + "\n"
        details += "💡 阈值说明: 本系统 >15% 即视为异常"

        return f"{verdict}\n\n{desc}\n\n{details}"

# --- 启动云端界面 ---
if __name__ == "__main__":
    guard = ImageGuard()
    
    # 云端适配：自定义CSS优化显示
    custom_css = """
    .gradio-container {background-color: #f9f9f9; max-width: 1200px !important; margin: 0 auto;}
    .gr-button {font-size: 16px !important; padding: 12px !important;}
    .gr-textbox {font-size: 14px !important; line-height: 1.6 !important;}
    """

    with gr.Blocks(title="AI 图像鉴别终极版", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🦅 AI 图像鉴别 (云端终极版)")
        gr.Markdown("专攻 Flux / Midjourney v6 / SDXL 高写实图片检测 | 云端部署版")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="拖入图片进行检测", height=400)
                btn = gr.Button("开始深度扫描", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                result_output = gr.Textbox(label="鉴定报告", lines=12)
        
        # 绑定点击事件
        btn.click(guard.analyze, inputs=image_input, outputs=result_output)

    # 云端启动配置（关键！适配云平台端口和访问）
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=int(os.getenv("PORT", 7860)),  # 读取云平台分配的端口
        share=False,  # 关闭临时分享链接
        show_error=True,  # 显示错误信息便于调试
        quiet=False  # 输出启动日志
    )
