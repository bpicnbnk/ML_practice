from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I come to China to travel",
          "This is a car polupar in China",
          "I love tea and Apple ",
          "The work is to write some papers in science"]
tfidf = TfidfVectorizer()
re = tfidf.fit_transform(corpus)
print(re)
