
import numpy as np



docs = [
    "https://www.baidu.com",
    "https://www.taobao.com",
    "https://www.jd.com",
    "https://www.1688.com",
    "https://www.12306.cn",
]

# vectors = [vectorizer.transform([doc]).toarray() for doc in docs]
# vectors = [[vectorizer.transform([docs[i]]).toarray()[0] for _ in range(3)] for i in range(len(docs))]
vectors = [vectorizer.transform([doc]).toarray()[0] for doc in docs]

print(vectors)
print("-----------------"*50)
data = [
    {"id": i, "doc": docs[i], "vector": vectors[i]}  
    for i in range(len(vectors))
]

print(len(vectors))

print(data)
