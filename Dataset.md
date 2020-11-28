# GLUE

- General Language Understanding Evaluation
- 9项
- ![img](https://pic2.zhimg.com/80/v2-2948db2c0d3a56b2282a01c954277d15_720w.jpg)



## MNLI

- [Multi-Genre Natural Language Inference](https://www.nyu.edu/projects/bowman/multinli/)
- 三分类，判断句子关系，众包数据集
- 组成
  - 433k sentence pairs annotated with textual entailment information
  -  modeled on the SNLI corpus
    - differs in that covers a range of genres of spoken and written text,support cross-genre generalization evaluation
  - 分为matched和mismatched两个版本的MNLI数据集，前者指训练集和测试集的数据来源一致，而后者指来源不一致
- served as the basis for the shared task of the [RepEval 2017 Workshop](https://repeval2017.github.io/shared/) at EMNLP in Copenhagen.

## QNLI

- [Question Natural Language Inference](https://www.nyu.edu/projects/bowman/glue.pdf)
- 二分类任务：（问题，句子）pair ， 句子是否有包含正确答案
- 组成
  - SQuAD数据

## RTE

- [Recognizing Textual Entailment](https://www.k4all.org/project/third-recognising-textual-entailment-challenge/)
- 二分类任务，类似于MNLI，但是只是蕴含或者不蕴含。训练数据更少

## WNLI

- [Winograd NLI](https://plmsmile.github.io/2018/12/15/52-bert/)
- 文本蕴含任务，不过似乎GLUE上这个数据集还有些问题



## CoLA

- [The Corpus of Linguistic Acceptablity](https://nyu-mll.github.io/CoLA/)
- 二分类任务，判断一个英语句子是否符合语法的



## SST-2

- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html)
- 二分类任务，给1个评论句子，判断情感
- SST-5是五分类，SST-5的情感极性区分的更细致

## MRPC

- [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
- 2分类任务，判断两个句子是否语义相等
  - 网上新闻组成
  - 05年的，3600条训练数据

## STS-B

- [The Semantic Textual Similarity Benchmark](http://alt.qcri.org/semeval2017/task1/)
- 多分类任务，判断两个句子的相似性，0-5
- 组成
  - 新闻标题和其他

## QQP

- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- 判断问题对是否有相同含义
- ![img](https://blog-10039692.file.myqcloud.com/1497495841438_1766_1497495841921.png)



# SWAG

- [The Situations With Adversarial Generations](https://rowanzellers.com/swag/) 
- 常识性推理数据集（阅读理解）
- 一个四分类问题。给一个陈述句子和4个备选句子，判断前者与后者中哪个最有逻辑的连贯性



# SQuAD

- #### Standford Question Answering Dataset

- 生成式任务

- 给一对语句

  - 问题
  - 一段来自wikipedia的文字，文字中隐含着问题的答案，答案是连续的

# CoNLL-2003 NER

- [数据集地址](https://www.clips.uantwerpen.be/conll2003/ner/)

- [推荐阅读](https://yuanxiaosc.github.io/2018/12/26/%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%ABCoNLL2003/)

- 组成
  - The first item on each line is a word, 
  - the second a part-of-speech (POS) tag, 
  - the third a syntactic chunk tag
  - the fourth the named entity tag.