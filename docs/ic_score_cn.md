## 总览

这份 JSON-Schema 定义了**界面复杂度（Interface Complexity, IC）**特征向量的三种记录形态： **工具级（tool）** 、 **服务器级（server）** 、 **工具集合级（set）** 。每条记录都包含：

* `features`：数值化特征（可扩展）。
* `components`：各特征经权重后的 **加和贡献** （便于归因/解释）。
* `score`：最终的 IC 得分（通常是线性模型/校准回归的输出）。

> 说明：下文出现的 `log()` 默认为自然对数（`ln`），除非你的仓库另有明确约定。

---

## 顶层字段

* **`$schema`** ：所用 JSON-Schema 版本（`draft-07`），校验时需要。
* **`$id`** ：此 Schema 的标识符/基准 URL，便于 `$ref` 解析和缓存。
* **`title`** ：人类可读的标题（“Interface Complexity Feature Vector”）。
* **`type`** ：顶层为 `"object"`。
* **`oneOf`** ：三选一的记录形态：`Tool IC entry` / `Server IC entry` / `Set IC entry`。
* **`definitions.weight_contributions`** ：通用引用类型，表示 **特征贡献字典** （`{ 特征名: 数值 }`）。常用来存放 `w_i * φ_i(x)` 的结果；可额外带 `"bias"` 键承载截距贡献。

---

## Tool IC entry（工具级记录）

 **语义** ：描述 **单个工具** （即某个可调用函数/工具接口）的复杂度。

### 标识/包络

* **`scope`** ：常量 `"tool"`。
* **`tool_id`** ：工具的稳定标识（如“serverA.get_user”）。
* **`server_id`** ：该工具所属服务器的标识，用于聚合/回溯。
* **`metadata`** （可选）：自由结构的溯源信息（提取时间、commit、文档 URL 等）。

### `features`（工具级特征）

> 下列特征默认从**工具参数 Schema（通常是 JSON-Schema/MCP 工具描述）**抽取。

* **`leaf_fields`**

  含义：叶子字段数量（非对象、非数组的终端字段，如 string/number/boolean/enum）。

  计算：遍历参数 AST，统计所有可直接赋值的叶节点。

  价值：直观反映“需要正确填的槽”数量。
* **`object_fields`**

  含义：`type: "object"` 的节点数量。

  价值：对象越多，层级结构和键约束越复杂。
* **`array_fields`**

  含义：`type: "array"` 的节点数量。

  价值：数组引入列表长度、元素模式等额外负担。
* **`total_fields`**

  含义：总体节点数（≈ `leaf_fields + object_fields + array_fields`）。

  价值：粗粒度复杂度上界。
* **`required_fields`**

  含义：被标记为必填的字段数量。

  价值：必填越多，失败成本/填充难度越大。
* **`required_ratio`**

  含义：`required_fields / max(leaf_fields, 1)`。

  价值：归一化的“约束密度”。
* **`max_depth`**

  含义：从根到任一叶子的 **最大嵌套深度** 。

  价值：层级越深，提示对齐与推理步骤越多。
* **`union_ops`**

  含义：出现的并合构造次数（`oneOf`/`anyOf`/`allOf` 等）。

  价值：每个并合都引入 **分支选择** ，增大决策空间。
* **`dependency_ops`**

  含义：字段依赖/条件约束的次数（`dependentRequired`、`if/then/else` 等）。

  价值：体现字段间 **耦合/互斥/前置条件** ，提升出错率。
* **`enum_bits`**

  含义：枚举选择的信息量总和。

  典型计算：对每个枚举字段计 `log2(|E|)`，求和。

  价值：工具调用时的 **离散选择负担** 。
* **`numeric_bits`**

  含义：数值字段的离散化选择负担。

  典型计算：若有 `min/max/step`，则 `choices≈floor((max−min)/step)+1`，计 `log2(choices)`；如仅有区间，上设默认分辨率近似。

  价值：连续值的**可行取值空间**负担。
* **`string_burden`**

  含义：自由文本生成难度的代理量。

  典型计算：未限定字符串计固定成本 + `log(maxLength)` + 正则/模式复杂度（如模式长度、字符类多样性）等的加权。

  价值：LLM 生成自由文本最易出错，需单列成本。
* **`array_cost`**

  含义：数组引入的额外负担。

  典型计算：对每个数组，加 `1 + log(expected_length+1)`，再乘以 **元素复杂度** （可用元素子树的 leaf/union 等近似）。`expected_length` 无约束时可取 1–3。

  价值：多元素/变长列表显著放大填充空间。
* **`doc_tokens`**

  含义：呈现给模型的工具文档/描述/参数注释的 **总 token 数** 。

  价值：直接影响提示长度与推理/选择延迟。
* **`missing_examples`**

  含义：缺失可用示例的惩罚。

  取值：可为二元（0/1）或“缺的示例数目”（如每参数+整例）。

  价值：无示例→更高探索成本/更高出错率。
* **`log_leaf_fields`**

  含义：`log(leaf_fields + 1)` 的稳定化版本。

  价值：抑制极端大工具对线性项的过度影响。

> **可扩展性** ：`"additionalProperties":{"type":"number"}` 允许随时新增实验特征（如 `regex_entropy`, `fewshot_quality`），无需改 Schema。

### 评分相关

* **`components`** ：各特征的**加权贡献**字典（通常是 `w_i * φ_i(x)` 的数值），可含 `"bias"`。用于可解释性与诊断。
* **`score`** ：工具级总分，常见形式：

  `score = Σ components + (bias)`

  分数越高，表示该工具**更难**被正确选择并填参。

---

## Server IC entry（服务器级记录）

 **语义** ：聚合 **单台服务器** （MCP server/HTTP 服务等）下所有工具的复杂度与“生态难度”。

### 标识

* **`scope`** ：常量 `"server"`。
* **`server_id`** ：服务器稳定标识。
* **`server_name`** （可选）：显示名称。
* **`tool_ids`** ：参与本次聚合的工具列表（便于溯源与下钻）。

### `features`（服务器级特征）

* **`tool_count`** ：该服务器的工具数量。
* **`log_tool_count`** ：`log(tool_count + 1)`，稳定极端值。
* **`mean_ic_tool`** ：该服务器下 **工具级 `score` 的平均值** 。反映“典型工具难度”。
* **`ambiguity`** ： **工具间可混淆度** 。

  典型：基于名称/描述/参数文档的向量相似度（如 top-k 余弦相似度均值），或参数名/语义重叠度。值越大，**选错工具**风险越高。

* **`naming_inconsistency`** ： **命名风格不一致度** 。

  典型：camelCase/snake_case/动词-名词风格的混用率、违反主流约定的比例、命名特征的方差等。值越大，认知负担越高。

* **`statefulness`** ： **有状态交互程度** 。

  典型：需要 session/cursor/pageToken 的工具比例；存在 begin/continue/commit 模式；文档中显式的前后条件。值越高，代理需 **记忆上下文** 。

* **`doc_burden`** ： **展示该服务器工具目录的 token 负担** 。可取工具 `doc_tokens` 之和，或“服务器概览 + Top-N 工具”的预算化 token。直接影响上下文花销与选择延迟。

### 评分相关

* **`components`** ：服务器级特征的加权贡献字典。
* **`score`** ：服务器级总分（越高越难）。

---

## Set IC entry（工具集合级记录）

 **语义** ：对“这一次实际**暴露给模型**的一组工具”的复杂度评估。用于提示设计、A/B、删减工具等实验。

### 标识

* **`scope`** ：常量 `"set"`。
* **`tools`** ：集合内工具的 `tool_id` 列表（可跨多服务器）。

### `features`（集合级特征）

* **`k`** ：集合大小（工具个数）。
* **`log_k`** ：`log(k + 1)`，稳定极端值。
* **`redundancy`** ： **集合内部冗余度** 。

  典型：工具文本嵌入的平均成对相似度；或能力标签的 Jaccard 重叠。冗余越高， **选择更难** 、上下文更浪费。

* **`token_load`** ：为呈现该集合（名称+简述+参数要点）所需的总 token。可对单工具展示做上限裁剪。
* **`mean_ic_tool`** ：集合内 **工具级 `score` 的均值** 。
* **`server_diversity`** ： **来源服务器多样性** 。

  典型：`|unique(server_id)| / k` 或标准化的多样性指数。多样性高可减少命名碰撞，但也可能增加风格异构负担。

### 评分相关

* **`components`** ：集合级特征的加权贡献字典。
* **`score`** ：集合级总分（越高表示这组工具整体更难被正确**挑选**与 **填参** ）。

---

## 实务建议（与仓库实现对齐）

1. **提取流程** ：解析工具 Schema → 遍历 AST 计数结构项 → 估算 `enum_bits`/`numeric_bits`/`string_burden`/`array_cost` → 统计 `doc_tokens`/`missing_examples` → 服务器与集合聚合 → 线性/校准模型打分 → 输出 `components` + `score`。
2. **可解释性** ：优先存 `components`（含 `bias`），便于仪表盘分析“究竟是什么拖高了分数”。
3. **可扩展性** ：利用 `additionalProperties` 快速加实验特征（如 `regex_entropy`、`example_coverage`）。

