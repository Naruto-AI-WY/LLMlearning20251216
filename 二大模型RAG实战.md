## **1 引言**

### **1.1 什么是RAG**

在上一节，我们成功搭建了一个源大模型智能对话Demo，亲身体验到了大模型出色的能力。然而，在实际业务场景中，通用的基础大模型可能存在无法满足我们需求的情况，主要有以下几方面原因：

- 知识局限性：大模型的知识来源于训练数据，而这些数据主要来自于互联网上已经公开的资源，对于一些实时性的或者非公开的，由于大模型没有获取到相关数据，这部分知识也就无法被掌握。
- 数据安全性：为了使得大模型能够具备相应的知识，就需要将数据纳入到训练集进行训练。然而，对于企业来说，数据的安全性至关重要，任何形式的数据泄露都可能对企业构成致命的威胁。
- 大模型幻觉：由于大模型是基于概率统计进行构建的，其输出本质上是一系列数值运算。因此，有时会出现模型“一本正经地胡说八道”的情况，尤其是在大模型不具备的知识或不擅长的场景中。
- 

为了上述这些问题，研究人员提出了检索增强生成（Retrieval Augmented Generation, **RAG** ）的方法。这种方法通过引入外部知识，使大模型能够生成准确且符合上下文的答案，同时能够减少模型幻觉的出现。

由于RAG简单有效，它已经成为主流的大模型应用方案之一。

如下图所示，RAG通常包括以下三个基本步骤：

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/f7f32892-9ace-4302-a8cd-169330bd053e.png)



- 索引：将文档库分割成较短的 **Chunk** ，即文本块或文档片段，然后构建成向量索引。
- 检索：计算问题和 Chunks 的相似度，检索出若干个相关的 Chunk。
- 生成：将检索到的Chunks作为背景信息，生成问题的回答。
- 

### **1.2 一个完整的RAG链路**

本小节我们将带大家构建一个完整的RAG链路。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/52b3dc6a-fb4b-4f98-95bd-9f265d7b399c.png)



（图片来源：https://github.com/netease-youdao/QAnything）

从上图可以看到，线上接收到用户 `query` 后，RAG会先进行检索，然后将检索到的 `Chunks` 和 `query` 一并输入到大模型，进而回答用户的问题。

为了完成检索，需要离线将文档（ppt、word、pdf等）经过解析、切割甚至OCR转写，然后进行向量化存入数据库中。

接下来，我们将分别介绍离线计算和在线计算流程。

#### **1.2.1 离线计算**

首先，知识库中包含了多种类型的文件，如pdf、word、ppt等，这些 `文档` （Documents）需要提前被解析，然后切割成若干个较短的 `Chunk` ，并且进行清洗和去重。

由于知识库中知识的数量和质量决定了RAG的效果，因此这是非常关键且必不可少的环节。

然后，我们会将知识库中的所有 `Chunk` 都转成向量，这一步也称为 `向量化` （Vectorization）或者 `索引` （Indexing）。

`向量化` 需要事先构建一个 `向量模型` （Embedding Model），它的作用就是将一段 `Chunk` 转成 `向量` （Embedding）。如下图所示：

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/9447d99a-dd80-4e51-bd36-3b294f510d0c.png)



一个好的向量模型，会使得具有相同语义的文本的向量表示在语义空间中的距离会比较近，而语义不同的文本在语义空间中的距离会比较远。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/667a4729-6fcf-4e48-ae43-77e5f4f915fc.png)



由于知识库中的所有 `Chunk` 都需要进行 `向量化` ，这会使得计算量非常大，因此这一过程通常是离线完成的。

随着新知识的不断存储，向量的数量也会不断增加。这就需要将这些向量存储到 `数据库` （DataBase）中进行管理，例如[Milvus](https://milvus.io/)中。

至此，离线计算就完成了。

#### **1.2.2 在线计算**

在实际使用RAG系统时，当给定一条用户 `查询` （Query），需要先从知识库中找到所需的知识，这一步称为 `检索` （Retrieval）。

在 `检索` 过程中，用户查询首先会经过向量模型得到相应的向量，然后与 `数据库` 中所有 `Chunk` 的向量计算相似度，最简单的例如 `余弦相似度` ，然后得到最相近的一系列 `Chunk` 。

由于向量相似度的计算过程需要一定的时间，尤其是 `数据库` 非常大的时候。

这时，可以在检索之前进行 `召回` （Recall），即从 `数据库` 中快速获得大量大概率相关的 `Chunk` ，然后只有这些 `Chunk` 会参与计算向量相似度。这样，计算的复杂度就从整个知识库降到了非常低。

`召回` 步骤不要求非常高的准确性，因此通常采用简单的基于字符串的匹配算法。由于这些算法不需要任何模型，速度会非常快，常用的算法有 `TF-IDF` ， `BM25` 等。

另外，也有很多工作致力于实现更快的 `向量检索` ，例如[faiss](https://github.com/facebookresearch/faiss)，[annoy](https://github.com/spotify/annoy)。

另一方面，人们发现，随着知识库的增大，除了检索的速度变慢外，检索的效果也会出现退化，如下图中绿线所示：

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/7d9df0e1-cce2-4de9-b2f0-fc684bc08f3c.png)



（图片来源：https://github.com/netease-youdao/QAnything）

这是由于 `向量模型` 能力有限，而随着知识库的增大，已经超出了其容量，因此准确性就会下降。在这种情况下，相似度最高的结果可能并不是最优的。

为了解决这一问题，提升RAG效果，研究者提出增加一个二阶段检索—— `重排` (Rerank)，即利用 `重排模型` （Reranker），使得越相似的结果排名更靠前。这样就能实现准确率稳定增长，即数据越多，效果越好（如上图中紫线所示）。

通常，为了与 `重排` 进行区分，一阶段检索有时也被称为 `精排` 。而在一些更复杂的系统中，在 `召回` 和 `精排` 之间还会添加一个 `粗排` 步骤，这里不再展开，感兴趣的同学可以自行搜索。1

综上所述，在整个 `检索` 过程中，计算量的顺序是 `召回` > `精排` > `重排` ，而检索效果的顺序则是 `召回` < `精排` < `重排` 。

当这一复杂的 `检索` 过程完成后，我们就会得到排好序的一系列 `检索文档` （Retrieval Documents）。

然后我们会从其中挑选最相似的 `k` 个结果，将它们和用户查询拼接成prompt的形式，输入到大模型。

最后，大型模型就能够依据所提供的知识来生成回复，从而更有效地解答用户的问题。

至此，一个完整的RAG链路就构建完毕了。

### **1.2 开源RAG框架**

目前，开源社区中已经涌现出了众多RAG框架，例如：

- [TinyRAG](https://github.com/KMnO4-zx/TinyRAG)：DataWhale成员宋志学精心打造的纯手工搭建RAG框架。
- [LlamaIndex](https://github.com/run-llama/llama_index)：一个用于构建大语言模型应用程序的数据框架，包括数据摄取、数据索引和查询引擎等功能。
- [LangChain](https://github.com/langchain-ai/langchain)：一个专为开发大语言模型应用程序而设计的框架，提供了构建所需的模块和工具。
- [QAnything](https://github.com/netease-youdao/QAnything)：网易有道开发的本地知识库问答系统，支持任意格式文件或数据库。
- [RAGFlow](https://github.com/infiniflow/ragflow)：InfiniFlow开发的基于深度文档理解的RAG引擎。
- ···
- 

这些开源项目各具优势，功能丰富，极大的推动了RAG技术的发展。

然而，随着这些框架功能的不断扩展，学习者不可避免地需要承担较高的学习成本。

因此，本节课将以 `Yuan2-2B-Mars` 模型为基础，进行RAG实战。希望通过构建一个简化版的RAG系统，来帮助大家掌握RAG的核心技术，从而进一步了解一个完整的RAG链路。

## **2 源2.0-2B RAG实战**

### 2.0 PAI实例创建

在实战之前，需要开通阿里云PAI-DSW试用，并在魔搭社区创建PAI实例，创建流程与速通手册一致~

如果忘记如何创建的，可以参考下面的内容复习一下~

#### [Step0：开通阿里云PAI-DSW试用](https://free.aliyun.com/?spm=5176.14066474.J_4683019720.1.8646754cugXKWo&scm=20140722.M_988563._.V_1&productCode=learn)

#### [Step1：在魔搭社区创建PAI实例！（点击即可跳转）](https://www.modelscope.cn/my/mynotebook/authorization)

### **2.1 环境准备**

进入实例，点击终端。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/8b8ae857-291b-471d-8aff-623a0b1cb666.png)



运行下面代码，下载文件，并将 `Task 3：源大模型RAG实战` 中内容拷贝到当前目录。

```
git lfs install
git clone https://www.modelscope.cn/datasets/Datawhale/AICamp_yuan_baseline.git
cp AICamp_yuan_baseline/Task\ 3：源大模型RAG实战/* .
```



![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/32bb6466-5eee-472d-acf9-1479921e06e6.png)



双击打开 `Task 3：源大模型RAG实战.ipynb` ，然后运行所有单元格。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/41ba0da0-12fc-4667-9adc-bc66f61225d5.png)



通过下面的命令，我们可以看到ModelScope已经提供了所需要的大部分依赖，如 `torch` ， `transformers` 等。

```
# 查看已安装依赖
pip list
```



![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/4f571546-a6d6-46a9-8201-2acdf84b0bb9.png)



但是为了进行模型微调以及Demo搭建，还需要在环境中安装 `streamlit` 。

```
# 安装 streamlit
pip install streamlit==1.24.0
```



安装成功后，我们的环境就准备好了。

### **2.2 模型下载**

在RAG实战中，我们需要构建一个向量模型。

向量模型通常采用BERT架构，它是一个Transformer Encoder。

输入向量模型前，首先会在文本的最前面额外加一个 `[CLS]` token，然后将该token最后一层的隐藏层向量作为文本的表示。如下图所示：

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/2b13148c-cc7c-4c8b-ab18-50e208550a40.png)



（在基于BERT的文本分类中，这个表示会送入分类器，得到标签。）

目前，开源的基于BERT架构的向量模型有如下：

- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)：智源通用embedding（BAAI general embedding, BGE）
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)：网易有道训练的Bilingual and Crosslingual Embedding
- [jina-embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh)：Jina AI训练的text embedding
- [M3E](https://huggingface.co/moka-ai/m3e-large)：MokaAI训练的 Massive Mixed Embedding
- ···
- 

除了BERT架构之外，还有基于LLM的向量模型有如下：

- [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)：智源LLM-Embedder
- ···
- 

其次，还有API:

- [OpenAI API](https://platform.openai.com/docs/guides/embeddings)
- [Jina AI API](https://jina.ai/embeddings/)
- [ZhipuAI API](https://open.bigmodel.cn/dev/api#text_embedding)
- ···
- 

在本次学习中，我们选用基于BERT架构的向量模型 `bge-small-zh-v1.5` ，它是一个4层的BERT模型，最大输入长度512，输出的向量维度也为512。

`bge-small-zh-v1.5` 支持通过多个平台进行下载，因为我们的机器就在魔搭，所以这里我们直接选择通过魔搭进行下载。

模型在魔搭平台的地址为[AI-ModelScope/bge-small-zh-v1.5](https://modelscope.cn/models/AI-ModelScope/bge-small-zh-v1.5)。

单元格 `2.2 模型下载` 会自动执行向量模型和源大模型下载。

- 首先是 **向量模型下载**

  ```
  # 向量模型下载
  from modelscope import snapshot_download
  model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
  ```

  

  这里使用的是 modelscope 中的 snapshot_download 函数，第一个参数为模型名称 `AI-ModelScope/bge-small-zh-v1.5` ，第二个参数 `cache_dir` 为模型保存路径，这里 `.` 表示当前路径。

  模型大小约为91.4M，由于是从魔搭直接进行下载，速度会非常快。

  下载完成后，会在当前目录增加一个名为 `AI-ModelScope` 的文件夹，其中 `bge-small-zh-v1___5` 里面保存着我们下载好的向量模型。

  ![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/5c03b38c-eea9-4437-8f28-278f78ca58b3.png)

- 另外，还需要 **下载源大模型**  `IEITYuan/Yuan2-2B-Mars-hf`

  下载方法和 `Task 1：零基础玩转源大模型` 一致。

  ```
  # 源大模型下载
  from modelscope import snapshot_download
  model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
  ```

  

  下载完成，如下图所示。

  ![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/22965361-415d-4d01-87ae-0ec987040d68.png)

- 

### **2.3 RAG实战**

模型下载完成后，就可以开始RAG实战啦！

#### 2.3.1 **索引**

为了构造索引，这里我们封装了一个向量模型类 `EmbeddingModel` ：

```
# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model = AutoModel.from_pretrained(path).cuda()
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List) -> List[float]:
        """
        calculate embedding for text list
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()
```



通过传入模型路径，新建一个 `EmbeddingModel` 对象 `embed_model` 。

初始化时自动加载向量模型的tokenizer和模型参数。

```
print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
embed_model = EmbeddingModel(embed_model_path)
```



`EmbeddingModel` 类还有一个 `get_embeddings()` 函数，它可以获得输入文本的向量表示。

注意，这里为了充分发挥GPU矩阵计算的优势，输入和输出都是一个 `List` ，即多条文本和他们的向量表示。

#### **2.3.2 检索**

为了实现向量检索，我们定义了一个向量库索引类 `VectorStoreIndex` ：

```
# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, doecment_path: str, embed_model: EmbeddingModel) -> None:
        self.documents = []
        for line in open(doecment_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)

        print(f'Loading {len(self.documents)} documents for {doecment_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist() 
```



类似地，通过传入知识库文件路径，新建一个 `VectorStoreIndex` 对象 `index` 。

初始化时会自动读取知识库的内容，然后传入向量模型，获得向量表示。

```
print("> Create index...")
doecment_path = './knowledge.txt'
index = VectorStoreIndex(doecment_path, embed_model)
```



上文提到 `get_embeddings()` 函数支持一次性传入多条文本，但由于GPU的显存有限，输入的文本不宜太多。

所以，如果知识库很大，需要将知识库切分成多个batch，然后分批次送入向量模型。

这里，因为我们的知识库比较小，所以就直接传到了 `get_embeddings()` 函数。

其次， `VectorStoreIndex` 类还有一个 `get_similarity()` 函数，它用于计算两个向量之间的相似度，这里采用了余弦相似度。

最后，我们介绍一下 `VectorStoreIndex` 类的入口，即查询函数 `query()` 。传入用户的提问后，首先会送入向量模型获得其向量表示，然后与知识库中的所有向量计算相似度，最后将 `k` 个最相似的文档按顺序返回， `k` 默认为1。

```
question = '介绍一下广州大学'
print('> Question:', question)

context = index.query(question)
print('> Context:', context)
```



这里我们传入用户问题 `介绍一下广州大学` ，可以看到，准确地返回了知识库中的第一条知识。

![图片](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/image/ed9fa75c-4fcc-4e93-a642-57d19e0c5bd2.png)



#### **2.3.3 生成**

为了实现基于RAG的生成，我们还需要定义一个大语言模型类 `LLM` ：

```
# 定义大语言模型类
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List):
        if context:
            prompt = f'背景：{context}\n问题：{question}\n请基于背景，回答问题。'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        output = self.tokenizer.decode(outputs[0])

        print(output.split("<sep>")[-1])
```



这里我们传入 `Yuan2-2B-Mars` 的模型路径，新建一个 `LLM` 对象 `llm` 。

初始化时自动加载源大模型的tokenizer和模型参数。

```
print("> Create Yuan2.0 LLM...")
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
llm = LLM(model_path)
```



`LLM` 类的入口是生成函数 `generate()` ，它有两个参数：

- `question` : 用户提问，是一个str
- `context` : 检索到的上下文信息，是一个List，默认是[]，代表没有使用RAG
- 

运行下面的代码，即可体验使用RAG技术之后 `Yuan2-2B-Mars` 模型的回答效果：

```
print('> Without RAG:')
llm.generate(question, [])

print('> With RAG:')
llm.generate(question, context)
```