## 分词

### `CountVectorizer()`

- [官网解释](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- 将文本中的词语转换为词频矩阵
- 通过fit_transform函数计算各个词语出现的次数。

```python
CountVectorizer(input='content', encoding='utf-8',  decode_error='strict', 
                strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
                stop_words=None,  token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), 
                analyzer='word', max_df=1.0, min_df=1, max_features=None, 
                vocabulary=None, binary=False, dtype=<class 'numpy.int64'>) 
```

- 实例

  - ```python
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    
    texts=['This is the first document.',
           'This is the second second document.',
           'And the third one.',
           'Is this the first document?'] 
    cv = CountVectorizer()
    cv_fit=cv.fit_transform(texts) #cv_fit 词频计数
    feature_name = cv.get_feature_name()
    print(cv.vocabulary_)
    # {'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': # 0, 'third': 7, 'one': 4}
    print(cv_fit.toarray())
    print(feature_name)
    # ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    ```

    

