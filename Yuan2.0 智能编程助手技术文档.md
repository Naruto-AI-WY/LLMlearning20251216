# Yuan2.0 智能编程助手技术文档

## 1. 项目概述

### 1.1 项目背景
随着软件开发和人工智能技术的发展，开发者们面临着日益复杂的编程挑战。传统的编程辅助工具往往只能提供基础的代码补全、错误提示等功能，难以理解开发者的整体意图或是上下文。

为了应对这一挑战，本项目利用最新的自然语言处理（NLP）技术，特别是 **浪潮源大模型 (Yuan2.0)**，开发了一款能够显著提升程序员工作效率的智能编程助手。

### 1.2 应用价值
* **提高开发效率**：减少重复工作和手动调试的时间，使开发者能更专注于业务逻辑和创新点。
* **降低学习成本**：帮助新手开发者更快地掌握新语言和技术栈。
* **增强代码质量**：辅助进行代码审查，确保代码遵循最佳实践。

---

## 2. 产品功能

本系统旨在为开发者提供全方位的编程辅助，主要功能包括：

1.  **智能代码生成**：根据用户自然语言描述，生成满足需求的代码片段。
2.  **多语言支持**：支持 Python, C++, Java, JavaScript 等主流编程语言。
3.  **代码理解和补全**：分析现有上下文，自动补全后续代码。
4.  **智能文档生成**：自动生成函数注释、模块说明文档。
5.  **代码优化建议**：识别低效代码并提供改进方案。

---

## 3. 技术方案与架构

### 3.1 方案架构
本项目采用 **Client-Server (C/S)** 架构模式（在本地部署演示中，前后端运行在同一环境中）。

* **服务端 (Model Backend)**：部署浪潮源 2.0 (Yuan2-2B) 大模型。负责加载模型权重、接收 Token 序列、执行推理计算并生成结果。
* **客户端 (Streamlit Frontend)**：基于 Python 的 Streamlit 框架开发 Web 交互界面。负责接收用户输入、维护对话历史 (Context)、将自然语言转换为模型可接受的 Prompt 格式，并将模型输出渲染展示。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/dashboard/dipwap/1763318810184/image.png)

### 3.2 数据流向

1.  **用户请求**：用户在 Streamlit 界面输入编程问题。
2.  **上下文拼接**：系统读取 Session State 中的历史对话，使用特定分隔符 `<n>` 进行拼接。
3.  **模型推理**：拼接后的 Prompt 输入到 Yuan2.0 模型，模型在 GPU 上进行计算生成。
4.  **结果返回**：模型输出的 Token 序列被解码为文本，并显示在前端。

---

## 4. 核心代码详细解析

以下是项目核心代码 `web_demo_2b.py` 的关键逻辑解释。

### 4.1 依赖导入与模型下载
利用 `modelscope` 解决大模型下载难、Git LFS 配置复杂的问题。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from modelscope import snapshot_download

# 使用 ModelScope 自动下载模型到本地缓存目录
# snapshot_download 会自动处理断点续传和本地缓存
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
path = './IEITYuan/Yuan2-2B-Mars-hf'
```

### 4.2 模型加载函数 (`get_model`)
这是系统的核心初始化部分。

```python
@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    # ------------------------------------------------------------------
    # 【关键配置】 use_fast=False
    # 原因：Yuan2.0 的 tokenizer.model 是 SentencePiece 二进制格式。
    # 新版 Transformers 默认尝试用 TikToken 解析会导致报错，
    # 必须显式禁用快速分词器，强制使用 Python 版 SentencePiece 加载。
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        add_eos_token=False, 
        add_bos_token=False, 
        eos_token='<eod>', 
        use_fast=False 
    )
    
    # 添加 Yuan2.0 特有的特殊 Token，用于代码补全和分隔
    tokenizer.add_tokens([
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', 
        '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>',
        '<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>',
        '<jupyter_output>','<empty_output>'
    ], special_tokens=True)

    print("Creat model...")
    # 加载模型权重
    # torch.bfloat16：在支持的 GPU (A10/3090) 上使用半精度以节省显存
    # trust_remote_code=True：允许加载模型仓库中的自定义代码
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).cuda()

    return tokenizer, model
```

### 4.3 交互与推理逻辑
Streamlit 的核心交互循环。

```python
# 获取用户输入
if prompt := st.chat_input():
    # 1. 记录并展示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. 构建 Prompt (提示词工程)
    # 使用 <n> 拼接历史对话，使用 <sep> 标记结尾
    prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
    
    # 3. 编码与推理
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    
    # do_sample=False：使用贪婪搜索 (Greedy Search)
    # 编程任务通常需要精确的逻辑，不建议使用随机采样
    outputs = model.generate(inputs, do_sample=False, max_length=1024)
    
    # 4. 解码与后处理
    output = tokenizer.decode(outputs[0])
    # 截取 <sep> 之后的内容作为本次回答
    response = output.split("<sep>")[-1].replace("<eod>", '')

    # 5. 展示回复
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
```

---

## 5. 环境配置与部署指南

为了确保代码能够稳定运行，避免 `TikToken` 解析错误或 `AttributeError`，请严格按照以下步骤配置环境。

### 5.1 基础依赖安装
安装运行所需的 AI 框架、Web 框架及工具库。

```bash
pip install torch streamlit modelscope sentencepiece einops accelerate
```

### 5.2 关键版本锁定 (Troubleshooting)
**注意**：由于 Yuan2.0 模型代码与新版 Transformers 存在兼容性问题（主要涉及 KV Cache 和 Protobuf 解析），**必须**锁定以下版本：

1.  **降级 Transformers** (解决 `AttributeError: NoneType...shape`):
    ```bash
    pip install transformers==4.30.2
    ```

2.  **降级 Protobuf** (解决 `TypeError: Descriptors cannot be created directly`):
    ```bash
    pip install protobuf==3.20.0
    ```

### 5.3 启动服务
在终端执行以下命令：

```bash
streamlit run web_demo_2b.py
```

启动成功后，控制台将输出访问地址（通常为 `http://localhost:8501`），在浏览器中打开即可开始使用。