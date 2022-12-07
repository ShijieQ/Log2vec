# Log2vec

```json
{
    "python" : 3.6
}
```

```shell
python graph_construct.py
python graph_embedding.py

```
```shell
conda install jupyter
conda install numpy
conda install matplotlib
conda install pandas
conda install tqdm
conda install -c anaconda scikit-learn
conda install tensorflow
conda install gensim
pip install fastdtw
pip install python-Levenshtein
```

## Node2vec流程
> 首先要明白的一点是，DeepWalk 是 q = 1 时 node2vev 的特殊情况，本项目中的deepwalk是狭义上的node2vev代码实现。

根据项目中的 `graph_emb\models\deepwalk.py`，我们大致理一下需要哪些步骤:
1. 生成一个带权的关注关系图，并获取图中节点，边以及对应的权重
   - 这部分在 `graph_embedding.py` 中导入之前由 `graph_contruct.py` 所生成的日志图
2. 为每一个点，每一条边生成基于 Node 和基于 Edge 的转移概率并生成 Alias Table
   - `graph_emb\walker.py` 中的 `preprocess_transition_probs()` 方法为我们做了这部分的工作
3. 根据epoch和walk length 以及 Alias Table 进行 Alias Sample 获取游走序列
4. 通过 Word2vec 方法对数据训练并获得向量 Embedding