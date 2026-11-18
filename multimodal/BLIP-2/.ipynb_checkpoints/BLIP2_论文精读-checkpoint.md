# BLIP-2：Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

- **原文**：arXiv:2301.12597v3 — [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)（本次精读所用版本即 arXiv v3 的 markdown 抽取稿，与你上传的 `2301.12597v3.md` 一致）
- **代码 / 项目**：[https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)（LAVIS 框架下的 BLIP-2 子项目，官方放出 OPT-2.7B / OPT-6.7B / FlanT5-XL / FlanT5-XXL 四档权重）
- **作者与机构**：Junnan Li、Dongxu Li、Silvio Savarese、Steven Hoi —— Salesforce Research
- **发表信息**：ICML 2023（PMLR v202）；arXiv 首发 2023-01-30，v3 为 2023 年修订版
- **一句话总结**：用 32 个可学习 query 把**冻结** ViT 的视觉特征压成 LLM 能直接"读"的软提示，先学表征（ITC/ITG/ITM 三目标、三种掩码）再学生成（prefix-LM / LM loss），只训一个 188M 的 Q-Former，就在零样本 VQAv2 上以 **54× 更少的可训练参数**超过 Flamingo-80B 8.7 分（65.0 vs 56.3）。
- **前置阅读**：BLIP（Li et al., 2022，本文的三目标与 CapFilt 直接沿用）→ CLIP（Radford et al., 2021，双塔对比与 ViT-L/14）→ Frozen（Tsimpoukelli et al., 2021，冻结 LLM + 软提示的开端）→ Flamingo（Alayrac et al., 2022，cross-attention 注入 + 交错图文数据，本文主要对手）→ BERT（Devlin et al., 2019，Q-Former 的初始化来源）→ UniLM（Dong et al., 2019，因果掩码的写法）

---

## 1. 研究背景与核心问题

### 1.1 当时的局面：VLP 越做越大，但成本越来越不可承受

2021–2022 年视觉-语言预训练（VLP）的路线是"端到端、越大越好"：ALBEF（210M）→ BLIP（583M）→ SimVLM（1.4B）→ BEIT-3（1.9B）→ Flamingo（80B）。论文开篇那句判断很直接：**pre-training 的成本已经 prohibitive**（难以承受），因为要端到端训练**大模型 + 大数据集**。

端到端的代价不只是钱，还有两个结构性问题：

1. **不灵活**：模型一旦端到端训好，视觉部分和语言部分就绑死了。想把语言端换成当年刚冒出来的 LLM（OPT、FlanT5），只能重训。
2. **浪费已有资产**：视觉社区已经有 CLIP/EVA-CLIP 这类高质量图像编码器，语言社区已经有千亿 token 训出来的 LLM。端到端 VLP 等于把这两笔资产都扔掉重新学一遍。

### 1.2 核心问题（本文自己提的三问）

论文的问题意识可以拆成三个层层递进的问题：

- **Q1（可行性）**：能不能**只训一个很小的中间模块**，把两侧的大模型都冻住，还做出 SOTA？
- **Q2（对齐）**：LLM 在单模态预训练时**从没见过图像**，把它冻住后，视觉特征怎么才能被它"看懂"？这是全篇的技术核心。
- **Q3（效率与可扩展）**：这套方法能不能做到"换更强的 encoder / 更强的 LLM → 更好的效果"，即把 VLP 变成一种**可收割单模态进展**的通用范式？

### 1.3 前人为什么不满足：生成损失不够

Frozen 与 Flamingo 都选了同一条捷径：**用图到文的生成损失（language modeling loss）** 把视觉特征压进 LLM。Frozen 微调图像编码器，让输出直接当软提示；Flamingo 往冻结 LLM 里插新的 cross-attention 层，在数十亿图文对上训这些新层。

论文对此的判断是本文最关键的"断言"（§1 第 3 段）：

> 现有方法依赖 image-to-text generation loss，**而我们证明这种损失不足以弥合模态鸿沟**（we show is insufficient to bridge the modality gap）。

这个断言不是空口，它有两处硬证据：

- **§4.1 "Effect of Vision-Language Representation Learning"（Fig 5）**：去掉第一阶段的表征学习、只留生成学习时，两类 LLM 的零样本 VQA 都显著下降；OPT 甚至出现**灾难性遗忘**——随着训练步数增加，性能持续下滑。
- **Table 6**：在检索任务上，只加生成式的 ITG 目标（ITC+ITM→ITC+ITM+ITG）也能涨点（COCO 微调 R@1：84.5→85.4、67.2→68.3；R@5：96.2→97.0、87.1→87.7），说明生成损失确实"学到了一点"对齐，但幅度远小于表征学习带来的整体跃迁。

于是 Q2 的答案变成了方法设计的出发点：**需要一个专门学"对准"的中间阶段，而不是让生成损失兼职干这件事。**

---

## 2. 核心方法与设计动机

### 2.1 一句话结构：Q-Former 是夹在两块冰之间的"温水瓶"

```
冻结 ViT (ViT-L/14 或 ViT-g/14)  ──►  Q-Former (188M, 可训练)  ──►  冻结 LLM (OPT / FlanT5)
        257×1024 视觉特征                32×768 查询表征 Z            Z 作为 32 个软提示 token
```

Q-Former（Querying Transformer）的四个设计要点，每一个都对应一个明确动机：

| 设计 | 具体做法 | 动机 |
|---|---|---|
| **可学习 query** | 32 个 query embedding，768 维，作为 image transformer 的输入 | 输出长度**固定**，与图像分辨率解耦（ViT-L/14 给 257 个 patch，ViT-g/14 更多，但 Z 恒定 32×768） |
| **共享 self-attention** | 图像 transformer 与文本 transformer **共用同一套 self-attn 层** | 三个目标共用一套参数，靠**不同掩码**切换任务，参数量最省 |
| **cross-attention 插在 Q-Former 内** | 每隔一个 transformer block 插入一个 cross-attn，query 从中读视觉特征 | 把"注入视觉"这件事从 LLM 里搬出来——Flamingo 是插进 LLM，本文是插在自己身上，因此**LLM 完全不用改结构**，可随意替换 |
| **BERT<sub>base</sub> 初始化** | self-attn 层用 BERT<sub>base</sub> 权重，cross-attn **随机初始化** | 借语言预训练的先验，同时不破坏 BERT 原来的单模态能力 |

一个容易被忽略的细节：**query 本身算模型参数**（论文明确注明）。所以 188M = BERT<sub>base</sub> 约 110M + cross-attn + query 的合计。

### 2.2 为什么是"瓶颈"，而不是"池化"

Z 的尺寸是 $32\times768$，而冻结 ViT-L/14 的特征是 $257\times1024$。张量元素数从 263K 降到 24.6K，**压缩约 10.7 倍**；若是 ViT-g/14（patch 更多、维度 1408），压缩比更大。

这个压缩不是副产品，而是核心机制。论文的说法是 "information bottleneck"：它一方面**逼着** query 只保留对文本有用的信息（因为容量不够装下全部视觉细节），另一方面**替 LLM 挡掉了**它不需要的视觉冗余——后者正是缓解灾难性遗忘的关键（§3.4 会给出形式化解释）。

### 2.3 两阶段预训练：先学"对准"，再学"说话"

- **Stage 1：视觉-语言表征学习（只接冻结 ViT，不接 LLM）**
  用图文对训练 Q-Former，三个目标同时优化：ITC（对比）、ITG（生成）、ITM（匹配）。这一阶段的目标不是"生成"，而是让 query 学会**抽取与文本最相关的视觉表征**。
- **Stage 2：视觉到语言的生成式学习（接冻结 LLM）**
  用 FC 层把 Z 投到 LLM 的词嵌入维度，**拼在文本嵌入前面**当软提示，只训 Q-Former（和那个 FC 层）。

为什么必须两阶段？反证法最清楚：如果只有 stage-2，Q-Former 就退化成 Flamingo 里的 Perceiver Resampler——它只能靠生成损失这个**稀疏、间接**的信号去学分对齐。论文在 §4.1 用 Fig 5 把这个反证做成了实验。

### 2.4 三种掩码：一套参数，三种"视角"

这是全文最精巧的设计。同一份 self-attention 权重，通过切换掩码，得到三种完全不同的信息流：

- **ITC（单模态掩码）**：query 只能看 query，文本只能看文本。**两路信息不许串门**——否则对比学习会"作弊"（文本侧提前看到了图像侧的表征，相似度计算就失去意义，即论文说的 information leak）。
- **ITG（多模态因果掩码）**：query 之间可互看、但**看不到任何文本 token**；第 $t$ 个文本 token 可看**全部 query** + **它之前的文本 token**。这条规则制造了本文最重要的信息学约束：文本 token 想获得视觉信息，**除了经由 Z 别无他途**。
- **ITM（双向掩码）**：全部可见，query 与文本自由互看，因此输出的 Z 里混着多模态信息，适合做"这对图文配不配"的二分类。

三张掩码图（我按论文 §3.1 与 Fig 2 的规则重绘，蓝色为可见）：

![Stage-1 三种注意力掩码](figs/masks.png)

---

## 3. 数学原理与完整推导

### 3.1 记号

设图像 $I$、文本 $Y=(y_1,\dots,y_n)$，冻结图像编码器 $f_{\phi}$（$\phi$ 不更新），冻结 LLM 记为 $\mathcal{L}_{\psi}$（$\psi$ 不更新），Q-Former 参数 $\theta$（唯一被更新的主体）。

$$
V = f_{\phi}(I)\in\mathbb{R}^{N_v\times d_v},\qquad Z = Q_{\theta}(V)\in\mathbb{R}^{K\times d},\quad K=32,\ d=768
$$

其中 $Z$ 由 32 个 query 的输出 $z_1,\dots,z_K$ 堆叠而成，$z_k\in\mathbb{R}^{768}$。文本侧得到 $T = \mathrm{TextEnc}_{\theta}(Y)$，取 `[CLS]` 位置作为句表征 $t\in\mathbb{R}^{768}$。

### 3.2 ITC：把 $K$ 个 query 压成一个相似度

**第一步，定义图文相似度。** 因为 $Z$ 有 $K$ 个向量而 $t$ 只有一个，论文的做法是"逐 query 算、取最大"：

$$
s(Z,t)=\max_{1\le k\le K}\ \frac{z_k^{\top}t}{\|z_k\|\,\|t\|}
$$

**第二步，InfoNCE。** 在一个 batch $\mathcal{B}$ 内，对每个正样本对 $(Z_i,t_i)$ 做 softmax 交叉熵（$\tau$ 为温度）：

$$
\mathcal{L}_{\mathrm{ITC}}=-\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\log\frac{\exp\big(s(Z_i,t_i)/\tau\big)}{\sum_{j=1}^{|\mathcal{B}|}\exp\big(s(Z_i,t_j)/\tau\big)}
$$

**第三步，看清梯度流向哪里。** $\max$ 的次梯度是"指示函数"：只有 $\arg\max_k$ 那个 query 会收到梯度，其余 31 个 query 在这一步**梯度为零**：

$$
\frac{\partial s}{\partial z_k}=\mathbb{1}\!\left[k=k^{\star}\right]\frac{1}{\|z_k\|\|t\|}\left(t-\frac{z_k^{\top}t}{\|z_k\|^2}z_k\right),\quad k^\star=\arg\max_k \frac{z_k^\top t}{\|z_k\|\|t\|}
$$

因此 ITC 实际上在做**硬注意力式的稀疏监督**：每个图文对只会"点亮"一个 query。这也解释了为什么必须配合 ITG/ITM —— 单靠 ITC，32 个 query 里大部分得不到充分训练，容易出现 query 塌缩（多个 query 学成同一模式）。

**第四步，负样本从哪来。** BLIP 用动量队列（momentum queue，队列长度可到万级），本文改用 **in-batch negatives**，原因写在 §3.2 里而且是纯工程逻辑：图像编码器是冻结的，所以单卡能塞更多样本（stage-1 batch 到 2320），负样本数就等于 $|\mathcal{B}|-1\approx2319$，比动量队列更"新鲜"且实现更简单。

代价要如实说：in-batch 负样本的分布偏向"同 batch 内其他图文对"，与真实负例分布有偏；且梯度方差随 batch 变化。这是用**实现简洁 + 显存换负样本质量**的取舍。

### 3.3 ITG：为什么"查询看不到文本"能逼出好表征

ITG 是标准的自回归语言建模，条件变量是 $Z$：

$$
\mathcal{L}_{\mathrm{ITG}}=-\sum_{t=1}^{n}\log p_{\theta}\big(y_t\mid y_{<t},Z\big)
$$

关键在于掩码施加的**结构性约束**：预测 $y_t$ 时，文本 token 能看到的是 $\{y_{<t}\}$ 与 $\{z_1..z_K\}$，**永远看不到 $y_{\ge t}$，也看不到图像 patch 本身**（图像 patch 与文本之间没有直接通路，只有 query→文本这一条）。于是：

$$
p_\theta(y_t\mid y_{<t},Z)=p_\theta(y_t\mid y_{<t},z_{1:K})\ \Longrightarrow\ I(Y;\ V\mid Z)\approx 0
$$

用信息论的话说：**$Z$ 被迫成为 $Y$ 关于 $V$ 的（近似）充分统计量**。这比 ITC 的监督信号强得多——ITC 只要求"排序对"，ITG 要求"能重建出整句话"，是对 $Z$ 容量的压力测试。这与"最小充分统计量"（minimal sufficient statistic）的思路同源：容量受限($32\times768$) + 重建要求严格 ⇒ 只能保留与文本相关的成分。

一个容易被问到的细节：`[CLS]` 被换成了新的 `[DEC]` token 作为第一个文本 token。原因是共享 self-attn 的前提下，"编码用 [CLS]、解码用 [DEC]"能让同一套权重区分两种模式，而不必再引入额外参数。

### 3.4 ITM：为什么"取平均"而不是"取最大"

ITM 是二分类：判断 $(I,Y)$ 是否匹配。做法是把每个 query 输出都送进一个二分类线性层，然后**对所有 query 的 logit 求平均**：

$$
s_{\mathrm{ITM}}(I,Y)=\sigma\!\left(\frac{1}{K}\sum_{k=1}^{K}w^{\top}z_k+b\right),\qquad
\mathcal{L}_{\mathrm{ITM}}=-\Big[y\log s+(1-y)\log(1-s)\Big]
$$

与 ITC 的 $\max$ 形成对照，这里是 $\mathrm{mean}$，动机很清楚：

- ITC 要对齐**一个**最相关的语义方向 ⇒ 用 $\max$ 选出最优点对点匹配；
- ITM 要判断**整体**是否匹配 ⇒ 用 $\mathrm{mean}$ 汇聚 32 个 query 的证据，方差更小；若用 $\max$，一个 query 的误判就能翻转结论，容易被"局部很像"的难负例骗过。

负例构造用 BLIP 的 **hard negative mining**：在一个 batch 里按图文相似度取最像但不匹配的样本作为负例，逼模型学细粒度差异（而不是靠"猫 vs 飞机"这种送分题）。

### 3.5 Stage 2：软提示的拼接与计算量的二次项

**投影。** 用 FC 层把 $Z$ 投到 LLM 词嵌入维度 $d_{\mathrm{LLM}}$：

$$
\tilde{Z}=ZW^{\top},\quad W\in\mathbb{R}^{d_{\mathrm{LLM}}\times 768}
$$

FlanT5-XXL 的 $d_{\mathrm{LLM}}=4096$，故 $W$ 仅 $768\times4096\approx3.1\mathrm{M}$ 参数——这就是 stage-2 唯一新增的参数。

**拼接（decoder-only，如 OPT）。**

$$
H^{(0)}=\big[\tilde{z}_1,\dots,\tilde{z}_{32};\ e(y_1),\dots,e(y_n)\big],\qquad
\mathcal{L}_{\mathrm{LM}}=-\sum_{t}\log p_\psi\big(y_t\mid \tilde{Z},y_{<t}\big)
$$

**拼接（encoder-decoder，如 FlanT5）：前缀语言建模。** 把 $Y$ 切成 $Y_p$（前缀）与 $Y_s$（后缀）：

$$
\mathcal{L}_{\mathrm{prefix\text{-}LM}}=-\sum_{t}\log p_\psi\big(y_{s,t}\mid \tilde{Z},Y_p,y_{s,<t}\big)
$$

即编码器输入 $[\tilde{Z};Y_p]$，解码器生成 $Y_s$。这个设计让 FlanT5 的编码器-解码器结构得以保留，而不是被强行改成 causal LM。

**为什么"瓶颈"同时也省算力（这一点论文没有明写，是我补的推导）。** 自注意力的计算量随序列长度二次增长。若把 257 个 patch 直接拼进 LLM，长度是 $257+n$；经 Q-Former 压成 32 个 token 后是 $32+n$。以 $n=32$ 为例：

$$
\frac{(257+32)^2}{(32+32)^2}=\frac{289^2}{64^2}\approx20.4
$$

即**注意力部分的算力降到约 1/20**。所以"信息瓶颈"不只是语义上的过滤器，也是一个**算力阀门**——这正是它能做到"16×A100(40G) 训最大模型 stage-1 < 6 天"的底层原因之一。

### 3.6 参数与算力账（含一处口径不一致）

| 项目 | 数值 | 出处 |
|---|---|---|
| Q-Former 总参数 | **188M** | §3.1 + Table 1（BLIP-2 行） |
| stage-2 新增 FC | $\approx3.1$M（$768\times4096$） | 由 FlanT5-XXL 的 $d=4096$ 推算 |
| 最大模型总参数 | 12.1B（ViT-g + FlanT5-XXL） | Table 2 |
| Table 2 记的可训练参数 | **108M** | Table 2 最后一行 |
| Table 1 记的可训练参数 | **188M** | Table 1 的 BLIP-2 行 |
| captioning 微调时可训练参数 | 1.1B（解冻 ViT-g） | Table 3 |
| VQA / 检索微调时可训练参数 | 1.2B（解冻 ViT-g） | Table 4 / Table 5 |

**这里有一处口径不一致必须点出来**：Table 2 写 BLIP-2（ViT-g + FlanT5-XXL）可训练 108M，而正文说 Q-Former 有 188M、Table 1 也写 188M。108M 恰好约等于 BERT<sub>base</sub> 的量级，**推测 Table 2 只统计了 self-attn/文本侧，漏算了 cross-attn 与 query**（约 80M）。这不影响结论（188M 仍然比 Flamingo-80B 的 10.2B 少 54×），但引用数字时要注明用的是哪张表的口径。

**算力估算（我用步数与 batch 反推，论文只给了天数）**：

$$
\text{stage-1 样本视图}\ \approx 250\mathrm{k}\times2320=5.8\times10^{8},\qquad
\text{stage-2}\ \approx 80\mathrm{k}\times1920=1.54\times10^{8}
$$

129M 张图 ⇒ stage-1 约 **4.5 个 epoch**，stage-2 约 1.2 个 epoch。折算到 16 卡、<6 天（约 $5.2\times10^5$ s）：

$$
\frac{5.8\times10^{8}}{5.2\times10^{5}}\approx1.1\times10^{3}\ \text{samples/s（全机）}\Rightarrow\ \approx70\ \text{samples/s/卡}
$$

这个量级与"ViT-g 冻结前向 + 188M Q-Former 反向 + 存 activation"是吻合的，反向传播的大头落在 Q-Former，而不是 1B 的 ViT-g。

### 3.7 灾难性遗忘为什么被缓解（机制层面的解释）

论文的说法是"Q-Former 作为信息瓶颈，减轻了 LLM 学对齐的负担，从而缓解灾难性遗忘"。把这句话翻译成可检验的因果链：

1. 冻结 ⇒ LLM 权重不变，其**已学到的语言分布**不会被打乱（这是最根本的一条，"遗忘"的前提被切断）。
2. 软提示长度固定为 32 个 token，输入分布的变化被限制在这 32 个位置上，**不会侵蚀原有的词嵌入空间**（对比 Frozen 那种微调整段图像编码器、让输出"入侵"prompt 空间的做法）。
3. stage-1 已把 Z 训练成"语言友好"的表征，stage-2 只需学一个线性映射 + 微调，梯度信号更干净。

Fig 5 里 OPT 在"无 stage-1"时性能随步数下降，正是第 2、3 条同时失效的后果：Q-Former 只能靠生成损失盲猜，输出分布持续漂移，把冻结 LLM 推到它没见过的输入区域。

---

## 4. 数据、训练与系统实现

### 4.1 数据

- **总量 129M 图文对**：COCO（Lin et al., 2014）、Visual Genome（Krishna et al., 2017）、CC3M、CC12M、SBU（Ordonez et al., 2011），以及 LAION400M 中的 **115M**。注意 115M 来自 LAION400M 的子集，不是全部（原文写 "115M images from the LAION400M dataset"）。
- **CapFilt 造描述**（沿用 BLIP）：对每张网络图，用 $\mathrm{BLIP}_{\mathrm{large}}$ 生成 **10 条**合成描述；连同 1 条原始网络描述共 11 条候选，用 **CLIP ViT-L/14** 的图文相似度排序，**保留 top-2**；每个预训练 step 从两条里**随机取一条**。随机取这一步很关键——它起到数据增强作用，避免模型只记住"最高分那条"的措辞。

### 4.2 冻结侧的两个选择

- **图像编码器**：CLIP 的 ViT-L/14 与 EVA-CLIP（Fang et al., 2022）的 ViT-g/14。
- **一个实用技巧**：**去掉 ViT 最后一层、取倒数第二层的输出**，效果略好。原因（我的解释）：最后一层是为 CLIP 的对比目标专门优化过的，语义更"贴合 CLIP 的文本空间"、也更抽象；倒数第二层保留了更细的视觉细节，对 VQA 这类需要局部证据的任务更友好。
- **LLM**：解码器类用无监督训练的 OPT 家族（2.7B / 6.7B），编码器-解码器类用指令微调的 FlanT5 家族（XL / XXL）。这个搭配本身就是一个对照实验：OPT vs FlanT5 的差异反映"指令微调 vs 无监督"的价值（Table 2 里 FlanT5-XL 在 VQAv2 上 62.3，而更大的 OPT-6.7B 只有 52.6）。

### 4.3 训练配置（照原文，含一处原文笔误）

| 项 | Stage 1 | Stage 2 |
|---|---|---|
| 步数 | 250k | 80k |
| batch（ViT-L / ViT-g） | 2320 / 1680 | OPT 1920 / FlanT5 1520 |
| 优化器 | AdamW，wd 0.05，$\beta=(0.9,0.98)$ | 同 |
| 学习率 | 峰值 $1\mathrm{e}{-4}$，余弦衰减，linear warmup 2k | 峰值 $1\mathrm{e}{-4}$，最小 $5\mathrm{e}{-5}$ |
| 精度 | FP16（FlanT5 用 BF16） | 同 |
| 输入 | 224×224，随机缩放裁剪 + 水平翻转 | 同 |
| 硬件 | 16×A100(40G)：最大模型 stage-1 < 6 天 | stage-2 < 3 天 |

**原文笔误**：§3.4 的优化器一行印成了 $\beta_1=0.9,\ \beta_1=0.98$（两个下标都是 1）。按 AdamW 的通行写法应为 $\beta_1=0.9,\beta_2=0.98$；中英对照稿里我加了括注说明，这里再次标出，复现时按 $(0.9,0.98)$ 取。

### 4.4 系统实现要点

- 全部跑在 **LAVIS** 框架内，Q-Former 与 ViT、LLM 通过统一的 `Blip2Base` 派生类拼装；stage-1 与 stage-2 是两个独立的 model 类（`Blip2QFormer` / `Blip2OPT` / `Blip2T5`）。
- 冻结侧用 `torch.no_grad()` + `.eval()`，但要保证 **ViT 的 BatchNorm/统计量不被更新**（EVA-CLIP 用的是 LayerNorm，无此问题；用其它 encoder 时要查清楚）。
- 混合精度下要注意 **FlanT5 的 BF16**：T5 系列在 FP16 下容易溢出（attention logits 与 log-softmax），论文明确对 FlanT5 用 BFloat16，这不是随手选的。

---

## 5. 实验设置与关键结论

### 5.1 三类零样本任务

1. **零样本 VQA**：OPT 用模板 `"Question: {} Answer:"`，FlanT5 用 `"Question: {} Short answer:"`（针对 T5 的输出习惯微调了模板，这点很容易漏）；生成用 beam width 5，**length penalty = −1**（鼓励短答案，更贴近人工标注的短答案分布）。
2. **零样本图到文生成**：直接在文本提示后接指令，靠 LLM 自身的指令跟随能力（Fig 4 的样例：解释产品卖点、判断照片反常、看图讲历史、写文案……）。
3. **零样本图文检索**：用**不带 LLM** 的第一阶段模型，直接在 COCO 上微调（ITC/ITM/ITG），再零样本迁到 Flickr30K。

### 5.2 关键数字（全部取自原文表格）

**零样本 VQA（Table 2，test-dev 口径）**

| 模型 | 可训练 | 总参数 | VQAv2 | OK-VQA | GQA |
|---|---|---|---|---|---|
| Frozen | 40M | 7.1B | 29.6 | 5.9 | – |
| VLKD | 406M | 832M | 44.5 | 13.3 | – |
| Flamingo-9B | 1.8B | 9.3B | 51.8 | 44.7 | – |
| **Flamingo-80B** | **10.2B** | **80B** | **56.3** | **50.6** | – |
| BLIP-2 ViT-g + OPT-2.7B | 107M | 3.8B | 52.3 | 31.7 | 34.6 |
| BLIP-2 ViT-g + OPT-6.7B | 108M | 7.8B | 52.6 | 36.4 | 36.4 |
| BLIP-2 ViT-g + FlanT5-XL | 107M | 4.1B | 63.0 | 40.7 | 44.2 |
| **BLIP-2 ViT-g + FlanT5-XXL** | **108M** | **12.1B** | **65.0** | **45.9** | **44.7** |

三个结论：

- **对 Flamingo-80B**：VQAv2 上 65.0 vs 56.3，**+8.7**；可训练参数 108M vs 10.2B，**约 94× 更少**（论文正文说 54×，用的是 188M 的口径：$10.2\mathrm{B}/188\mathrm{M}\approx54$）。两个数字都对，取决于用哪张表的 188M/108M——这也再次说明参数口径要交代清楚。
- **OK-VQA 输了**：45.9 vs 50.6。论文的归因是"OK-VQA 更依赖开放世界知识，而 Flamingo 背后的 70B Chinchilla 知识量超过 11B FlanT5-XXL"。这是一个诚实的归因，不是找借口：OK-VQA 需要的是外部知识检索，不是更强的视觉对齐。
- **同参数下 FlanT5 完胜 OPT**：FlanT5-XL（4.1B 总参）63.0，远高于 OPT-6.7B（7.8B 总参）52.6。**指令微调带来的"会按格式答题"的能力，比纯粹堆 LLM 参数更值钱。**这是全文最实用的一条经验。

**参数效率全景**（我按 Table 2 画的散点，横轴可训练参数取对数）：

![可训练参数 vs 零样本 VQAv2](figs/params_vs_vqa.png)

BLIP-2 的点几乎垂直贴在 $10^{8}$ 附近，把整条 Pareto 前沿**向左推了一个数量级**。

**图像描述（Table 3，NoCaps val / COCO 微调）**

| 模型 | 可训练 | NoCaps in-dom C | out-dom C | COCO CIDEr |
|---|---|---|---|---|
| BLIP | 446M | 114.9 | 115.3 | 136.7 |
| SimVLM | ~1.4B | 113.7 | 115.2 | 143.3 |
| BLIP-2 ViT-g + OPT-2.7B | 1.1B | 123.0 | 123.4 | 145.8 |
| BLIP-2 ViT-g + FlanT5-XL | 1.1B | 123.7 | 124.8 | 144.5 |

NoCaps 的域外（out-domain）CIDEr 从 115.3 提到 124.8，**+9.5**，这是"泛化到没见过的图像域"的直接证据。注意此表可训练参数是 **1.1B**，因为 captioning 微调**解冻了图像编码器**（ViT-g 约 1B + Q-Former 188M）；VQA 与检索微调同理，Table 4/5 记作 **1.2B**。换句话说：**"冻结 ViT"只属于预训练阶段，到了下游微调就解冻了**——这一点在论文正文里只出现在 §4.3 的一句话，很容易被读漏。

**图文检索（Table 6 的 ITG 消融，COCO 微调）**

| 目标 | I→T R@1 | I→T R@5 | T→I R@1 | T→I R@5 |
|---|---|---|---|---|
| ITC + ITM | 84.5 | 96.2 | 67.2 | 87.1 |
| ITC + ITM + ITG | **85.4** | **97.0** | **68.3** | **87.7** |

**一个反直觉但自洽的结果**：检索本来不需要"生成"，但加上生成目标反而涨点。原因正是 §3.3 的推导——ITG 把 Z 压成了 Y 的充分统计量，让 ITC/ITM 用的特征本身更"语言对齐"。

---

## 6. 消融实验与失败案例

### 6.1 消融 1：去掉 stage-1（本文最核心的消融）

Fig 5 对比"有/无表征学习"两种设定下零样本 VQAv2 随训练步数的曲线：

- **FlanT5**：无 stage-1 时起点低、涨得慢，最终明显低于有 stage-1 的版本；
- **OPT**：无 stage-1 时**性能随步数持续下降**——这是灾难性遗忘的教科书式表现：冻结 LLM 在持续接收分布漂移的软提示，原有的语言能力被破坏，而生成损失又不足以把新分布拉回"LLM 熟悉"的区域。

这条曲线是全文最有说服力的证据，因为它把"生成损失不够"从断言变成了实验事实。

### 6.2 消融 2：encoder × LLM 的四象限（Table 2 内部对比）

把 Table 2 的 8 个配置按两个维度切开，可以读出稳定的单调性：

| | OPT-2.7B | OPT-6.7B | FlanT5-XL | FlanT5-XXL |
|---|---|---|---|---|
| ViT-L | 49.7 | – | 62.3 | – |
| ViT-g | 52.3 | 52.6 | 63.0 | **65.0** |

两条单调性同时成立：**换更强的 encoder 涨点**（49.7→52.3，+2.6；62.3→63.0，+0.7），**换更大的 LLM 涨点**（52.3→52.6；63.0→65.0）。这就是 §1 里 Q3 的答案：BLIP-2 是一个能"收割"单模态进展的通用范式。

### 6.3 失败案例（原文 §5 + Fig 6）

论文自己列了三类失败，Fig 6 给了图示：

1. **LLM 的知识不准**（模型被冻结，它知道的就那么多，且可能是错的）；
2. **激活了错误的推理路径**（视觉证据够了，但推理链走歪）；
3. **缺少最新信息**（新出现的图像内容/实体，冻结的 LLM 不可能知道）。

另外两条"能力边界"式失败：

- **上下文学习（in-context learning）失效**：给 LLM 塞几个 VQA 示例，性能不涨。归因很具体——预训练数据每个样本**只有一对图文**，LLM 学不到"一条序列里多对图文之间的关联"。Flamingo 能用交错数据（M3W）做到，BLIP-2 的数据形态天然不支持。
- **继承 LLM 的风险**：冒犯性语言、社会偏见、隐私泄漏。缓解办法：指令引导 + 数据过滤。

---

## 7. 方法局限与适用边界

按"这套方法在什么条件下成立"来划，比列缺点更有用：

| 边界 | 具体限制 | 何时会踩到 |
|---|---|---|
| **容量边界** | Z 固定 32×768，容量硬上限 | 需要密集细粒度证据的任务：OCR、图表问答、小目标计数、长文档版面理解 |
| **数据形态** | 预训练只有单图文对，无交错序列 | 需要 few-shot in-context 的场景（论文已验证失效） |
| **知识边界** | LLM 冻结 ⇒ 知识停在训练时点 | 时效性问题、长尾实体 |
| **训练目标边界** | 只有 ITC/ITG/ITM，没有 grounding / detection 类监督 | 需要定位框、需要像素级指代的任务 |
| **任务边界** | 只验证了 VQA / captioning / retrieval 三类 | 视频、音频、多图推理均未涉及 |
| **可复现边界** | 129M 数据 + 16×A100，个人难以全量复现 | 学术小实验室只能复现 stage-2 或小规模 stage-1 |

还有一条**论文没写、但工程上真实**的边界：Q-Former 与 LLM 的**分词器耦合**。stage-2 的软提示是 32 个向量，它们不走 tokenizer，但 prompt 模板（`"Question: {} Answer:"` vs `"Question: {} Short answer:"`）严重依赖具体 LLM 的输出习惯——换 LLM 就得重调模板，这是"通用范式"里不那么通用的一面。

---

## 8. 与其他论文的演进关系

```
CLIP (2021)          双塔对比，ViT-L/14 —— 提供"冻结视觉"的资产
   │
ALBEF/BLIP (2021-22) ITC+ITM+LM 三目标、CapFilt、hard negative
   │                 —— BLIP-2 的 stage-1 几乎是它的"只训小模块"版本
Frozen (2021)        微调图像编码器 → 软提示进冻结 LLM
   │                 —— 证明了"冻结 LLM 可行"，但代价是解冻编码器
Flamingo (2022)      插入 cross-attn 到冻结 LLM、Perceiver Resampler、交错数据
   │                 —— BLIP-2 的直接对手；本文用"cross-attn 放自己身上"换来了可换 LLM
   ▼
BLIP-2 (2023)        Q-Former + 两阶段 + 表征学习先行
   │
   ├─► InstructBLIP (2023)      instruction-aware query（把指令也送进 Q-Former）
   ├─► LLaVA (2023)            弃用 Q-Former，ViT patch + 单一线性投影，并微调 LLM（LoRA/全参）
   ├─► MiniGPT-4 (2023)        反过来"吃现成"：直接用 BLIP-2 预训练好的 ViT+Q-Former，只训一个线性层
   ├─► Qwen-VL / InternVL (2023-24) 动态分辨率 + 更多视觉 token，绕开 32 query 的容量上限
   └─► Idefics / 各类 resampler 系 继续用"固定 query 数"控制视觉 token 预算
```

三个值得记住的演进判断：

1. **BLIP-2 的"必须两阶段"这个结论，后来被 LLaVA 类方法部分推翻。** LLaVA 用**一个线性层** + 解冻（或 LoRA）LLM 的指令微调，效果也很好。差别在于：LLaVA 允许 LLM 更新（不存在灾难性遗忘这个约束，因为它在指令数据上重新对齐），BLIP-2 把"LLM 完全冻结"当成硬约束，才需要 stage-1 来保护。**换句话说，stage-1 的收益是"冻结前提下的对齐效率"，不是绝对性能上限。**
2. **32 query 的容量上限是后来被绕开的第一个点。** Qwen-VL、InternVL 走向"更多视觉 token + 动态分辨率"，本质上承认了信息瓶颈在密集任务上不够用。
3. **但"控制视觉 token 预算"这件事被保留下来了。** 所有省算力的 MLLM 都在做类似的事：不是要不要压，而是压多少、怎么压、能不能动态压。

---

## 9. 复现记录与工程踩坑

**先交代记录的性质**：这一节我如实区分「我实际跑过的」与「按论文与官方代码推断的」。

- ✅ **实际做过**：本仓库内对 arXiv v3 全文做了 markdown 结构化、逐段中英对照（`blip2_paper/2301.12597v3_中英对照.md`），核对了全部表格数值与掩码规则，并按 §3.1/Fig 2 重绘了三张掩码示意图与参数效率散点（`blip2_paper/figs/`，脚本 `figs_make.py`，纯 matplotlib，未用实测数据、全部取自论文表格）。
- ⚠️ **没有实跑**：没有训练或推理 BLIP-2 权重。因此下面关于权重、显存、踩坑的内容是**推断 + 官方实现约定**，不是实测日志，请按这个前提看。

### 9.1 复现路径（按 LAVIS 官方流程）

```bash
# 1) 环境：LAVIS + transformers（BLIP-2 在 LAVIS 的 blip2 子项目里）
git clone https://github.com/salesforce/LAVIS.git && cd LAVIS && pip install -e .

# 2) 直接用现成权重做零样本推理（最小验证路径，单卡即可）
python -m lavis.projects.blip2.eval_vqa  # 官方脚本；权重名如 blip2_opt_2.7b / blip2_flan_t5_xxl
```

### 9.2 踩坑清单（按"最容易浪费一天"的顺序排）

| # | 坑 | 具体表现 | 处理 |
|---|---|---|---|
| 1 | **Table 2 的 108M vs Table 1 的 188M** | 引用"少 54×"还是"少 94×"会打架 | 统一口径：Q-Former 全参 188M（含 cross-attn 与 query）；108M 疑为漏算 |
| 2 | **BERT<sub>base</sub> 初始化 + cross-attn 随机** | 若把 cross-attn 也当预训练权重加载，会静默 load 失败/全零 | 只 load `bert-base-uncased` 的 self-attn/FFN 部分，cross-attn 显式随机初始化 |
| 3 | **ViT 取倒数第二层** | 用最后一层能跑通但掉点（论文说"略好"） | 在 encoder forward 里取 `hidden_states[-2]`，并在冻结点上确认不误取 CLS/pooled 输出 |
| 4 | **FlanT5 必须 BF16** | FP16 下 T5 的 attention/log-softmax 易溢出，loss 变 nan | 分支设置 `torch.bfloat16`（论文明确如此），别一刀切 FP16 |
| 5 | **stage-1 batch 2320 的显存** | 单卡放不下 145 样本/卡（16 卡） | gradient checkpointing + 冻结 ViT 的 `no_grad` + 累积步；或按比例缩 batch 并调 lr |
| 6 | **ITM 的 hard negative 需要跨卡负例** | 单卡 batch 小 → 负例不足 → ITM 学不到细粒度 | 用 all-gather 拼负例池（等价于论文的 in-batch 思路） |
| 7 | **beam search 的 length penalty = −1** | 默认 `length_penalty=1.0` 会偏向长答案，VQA 掉点明显 | 生成时显式设 `length_penalty=-1.0, num_beams=5`，并按 OPT/FlanT5 切换模板 |
| 8 | **CapFilt 数据准备极慢** | BLIP<sub>large</sub> 对 1.15 亿张图生成 10 条描述 | 先缓存 top-2 结果（离线一次性），训练时只读缓存；否则复现卡在数据阶段 |
| 9 | **captioning 微调解冻 ViT** | Table 3 的可训练是 1.1B，与 VQA 的 108M 不同 | 按任务改 `requires_grad`：captioning 解冻 ViT，VQA/检索冻结（VQA 只训 Q-Former + 微调图像编码器，见 §4.3 原文） |
| 10 | **prompt 模板与分词器** | OPT 与 FlanT5 的空格/token 边界不同 | 严格照原文两个模板，别统一成一个 |

第 9 条值得展开，因为它牵涉"到底冻哪些参数"这个最容易搞错的地方。原文 §4.3（VQA）写的是"finetune the parameters of the Q-Former **and the image encoder** while keeping the LLM frozen"，而 Table 4 给出的 BLIP-2 VQA 微调可训练参数是 **1.2B**（ViT-g 约 1.0B + Q-Former 188M），与文字**完全自洽**——也就是说：**"冻结 ViT"只发生在预训练阶段，一旦转到下游微调（captioning / VQA / 检索），图像编码器就解冻了。**

于是全文一共有四种"可训练参数"口径，引用时必须说清阶段：

| 阶段 / 任务 | 可训练参数 | 冻结的部分 |
|---|---|---|
| 预训练（Table 1 口径） | 188M（Q-Former 全参） | ViT + LLM |
| 预训练（Table 2 口径） | 108M（疑漏算 cross-attn/query） | ViT + LLM |
| Captioning 微调（Table 3） | 1.1B | LLM（ViT 解冻） |
| VQA / 检索微调（Table 4 / 5） | 1.2B | LLM（ViT 解冻） |

顺带一个跨表的微小不一致：captioning 记 1.1B、VQA 与检索记 1.2B，而三者解冻的东西应当相同（ViT-g + Q-Former）。这多半是四舍五入或统计范围差一点，不影响结论，但**别把 1.1B 和 1.2B 当成两种不同的解冻策略**——我一开始就误以为 VQA 只解冻了几层，去核对表格才发现是自己想多了。

---

## 10. 个人思考与后续问题

### 10.1 三个我认为最值得追问的点

**（1）"32 个 query"这个数是拍出来的，全文没有 query 数的消融。**
32×768 的信息量上限是设计出来的硬约束，但论文只做了 encoder/LLM 的二维扫描，没做 $K\in\{16,32,64,128\}$ 的扫描。这意味着我们无法从论文判断：**性能是"瓶颈逼出了好表征"，还是"32 恰好够用"**。对做工程的启发是——如果你的下游偏密集（OCR、图表），别信 32，直接上 64/128 或换动态分辨率；如果是粗粒度语义（caption、检索），32 可能都嫌多。

**（2）stage-1 的价值到底在"对齐"还是在"数据效率"？**
论文把它解释为"弥补生成损失不足"，但 LLaVA 用一个线性层也做成了。我的判断是：stage-1 的真正作用是**在 LLM 完全冻结的前提下，把对齐工作前置到一个便宜的模块里**。一旦允许微调 LLM（哪怕 LoRA），stage-1 的必要性就大幅下降。这个判断有可检验的推论：如果做"BLIP-2 + LoRA 微调 LLM"的对照，我预期 stage-1 带来的增益会明显缩小——这是我下一步真想跑的实验。

**（3）信息瓶颈 = 算力阀门，这个双重身份值得单独设计。**
我在 §3.5 推出的 $\approx20\times$ 注意力算力节省，论文只当副产品。但换个视角：**视觉 token 数应该随任务难度动态变化**——简单图像用 16 个 token，密集文档用 256 个。Q-Former 的 query 数是静态的，这正是后来 Qwen-VL / InternVL 用"动态分辨率 + 可变 token 数"要解决的问题。谁把"动态 $K$"做进 Q-Former，谁就同时拿到了算力和精度。

### 10.2 与我自己的手写 Llama3.1 项目怎么接

我正在 `ai-from-scratch` 里手写 Llama3.1 全流程，BLIP-2 这套结构对它有直接可借的两点：

1. **"跨模块只用一个 FC"的极简接口**。BLIP-2 的 stage-2 只在 Z 与 LLM 之间加一个 $768\times d_{\mathrm{LLM}}$ 的线性层，其余全靠训练 Q-Former。这对手写实现极友好：我的 Llama 侧一行代码都不用改（只需支持"外部传入 prefix embedding"），多模态能力就挂上去了。相比 Flamingo 要插 cross-attn 层（等于改模型结构、改 KV cache 逻辑），这条路线对我这种"想看清每一层"的目标更合适。
2. **prefix embedding 与 KV cache 的交互**。软提示是 32 个 token 的前缀，它们的 KV 在推理时是**恒定**的——可以在第一次 forward 时算好并复用。这一点在我手写 KV cache 时是天然的省算力点：图像编码 + Q-Former 只跑一次，32 个位置的 K/V 缓存下来，后续每轮对话只增量算文本 token。
3. **一个需要小心的坑**：我的 Llama 用的是 RoPE。软提示是拼在最前面的 32 个位置，会**整体挤占位置编号**（后续文本的 position id 从 32 开始）。这会让模型看到"文本从第 33 个位置开始"的偏移，与纯文本预训练时的分布不一致。BLIP-2 的 OPT（也带位置嵌入）实际就吃了这个偏移，只是靠预训练吸收了。若我要做，会考虑给软提示**单独的位置区间**或做 position id 的连续性校正。

### 10.3 还没解决的问题（留给下一篇笔记）

- **OK-VQA 的知识缺口怎么补？**论文的答案是"换更大的 LLM"，但更合理的路是接外部检索（RAG）。BLIP-2 的 32 个 query 恰好可以当检索的 query 表示——这条线在后续的 REVIVE、以及各类"视觉检索增强"工作里被走过了，值得单独精读。
- **Fig 5 的曲线我只读了定性形状**（OPT 随步数下降、FlanT5 平台更低）。具体每一步的数值需要看原图的高清版或官方 log，本文档不编数。
- **InstructBLIP 只加了一个"指令感知 query"就显著提升了指令跟随**，说明 BLIP-2 的 Q-Former 里还有明显的"指令信息没进去"的空隙——这是 §7 那个"数据形态边界"之外，第二个可以补的短板。

### 10.4 一句话收束

BLIP-2 的价值不在"65.0 这个数字"，而在于它把 VLP 从"训一个大家伙"改写成了**"训一个小接口，去对接两个已经很强的大件"**，并且用"信息瓶颈"这一个设计同时解决了对齐、遗忘、算力三件事。这种"用结构约束替代数据与算力堆叠"的思路，比它的具体分数更耐用。
