import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data_path = "./data/"
data_file_name = "originData.csv"

if __name__ == "__main__":

    origin_df = pd.read_csv(data_path+data_file_name, encoding="latin")
    origin_df.columns = ['string_label', 'text']
    num_label = []
    for index, row in origin_df.iterrows():
        if row['string_label'] == "ham":
            num_label.append(0)
        else:
            num_label.append(1)
    origin_df['num_label'] = num_label
    print(origin_df.head())

    vectorizer = TfidfVectorizer(binary=True)
    x = vectorizer.fit_transform(origin_df['text'])
    y = origin_df['num_label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
    print("训练数据中的样本个数: ", X_train.shape[0], "测试数据中的样本个数： ", X_test.shape[0])

    clf = MultinomialNB(alpha=1, fit_prior=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("测试数据集的准确度： ", accuracy_score(y_test, y_pred))