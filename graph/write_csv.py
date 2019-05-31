import json
import csv
import jieba
import random

docs, doc_content = json.load(open('../data/doc_content.json'))
field_names = ['url', 'label', 'content', 'concepts', 'title']
train_csv_writer = csv.DictWriter(open('../data/train_graph_data.csv', 'w'), fieldnames=field_names,
                                  delimiter='|', quotechar='\"', quoting=csv.QUOTE_ALL)
dev_csv_writer = csv.DictWriter(open('../data/dev_graph_data.csv', 'w'), fieldnames=field_names, delimiter='|',
                                quotechar='\"', quoting=csv.QUOTE_ALL)
test_csv_writer = csv.DictWriter(open('../data/test_graph_data.csv', 'w'), fieldnames=field_names, delimiter='|',
                                 quotechar='\"', quoting=csv.QUOTE_ALL)
train_csv_writer.writeheader()
dev_csv_writer.writeheader()
test_csv_writer.writeheader()
urls = set(docs.keys()).intersection(set(doc_content.keys()))
dev_docs = set(random.sample(urls, 500))
test_docs = set(random.sample(urls - dev_docs, 500))
train_docs = urls - dev_docs - test_docs

for url in urls:
    samples = docs[url]
    if url in doc_content:
        content = doc_content[url]
    else:
        continue
    title, abstract, tags, topic, _ = samples[0]
    comments = [s[-1] for s in samples]
    if url in train_docs:
        for sample in samples:
            title, abstract, tags, topic, comment = sample
            # comment[0] is the text of comment, comment[1] is the rating score
            train_csv_writer.writerow({'url': url, 'label': ' '.join(comment[0]), 'content': ' '.join(content),
                                       'concepts': ','.join([t[0] for t in tags if float(t[1]) > 0]),
                                       'title': ' '.join(title)})
    elif url in dev_docs:
        dev_csv_writer.writerow(
            {'url': url, 'label': '$$'.join([' '.join(comment[0]).replace('$', '') for comment in comments]),
             'content': ' '.join(content), 'concepts': ','.join([t[0] for t in tags if float(t[1]) > 0]),
             'title': ' '.join(title)})
    elif url in test_docs:
        test_csv_writer.writerow(
            {'url': url, 'label': '$$'.join([' '.join(comment[0]).replace('$', '') for comment in comments]),
             'content': ' '.join(content), 'concepts': ','.join([t[0] for t in tags if float(t[1]) > 0]),
             'title': ' '.join(title)})
    else:
        print('error! url not in doc contents')
