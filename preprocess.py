import jieba
import json
import os


def process_sample(line):
    term = line.strip().split('\t')
    if len(term) != 9:
        return None
    url = term[0]
    tags, topic = process_tags(term[4])
    for tag in tags:
        jieba.add_word(tag[0])  # tag: (tag, value)
    # title = list(jieba.cut(term[1]))
    title = term[1]
    # abstract = list(jieba.cut(term[2]))
    abstract = term[2]
    # comments = process_comments(term[8])
    comments = term[8]
    return url, title, abstract, tags, topic, comments


def process_tags(tags):
    terms = tags.split('$')[:-1]  # line ends with $
    tags = []
    topics = []
    for term in terms:
        tem = term.split(':')
        if len(tem) != 3:
            continue
        tag, type, value = tem
        if float(value) >= 0 and type == 'tag':
            tags.append((tag, float(value)))
        elif type == 'topic':
            topics.append(tag)
    topic = None
    if len(topics) > 0:
        topic = topics[0]
        for i in range(len(topics)):
            if len(topics[i]) > len(topic):
                topic = topics[i]
        topic = topic.split('_')
    return tags, topic


def process_comments(comments):
    terms = comments.split('$$')  # line ends with $$
    comments = []
    for term in terms:
        tem = term.split('::')
        if len(tem) != 2:
            print(tem)
            continue
        comment, value = tem
        try:
            value = float(value)
        except ValueError:
            print(tem)
            tem = term.split(':::')
            if len(tem) != 2:
                print(tem)
                continue
            comment, value = tem
        comment_words = list(jieba.cut(comment))
        try:
            value = float(value)
        except ValueError:
            print(term)
            continue
        comments.append((comment_words, value))
    return comments


def read_file(fname, topic_limit=[u'娱乐']):
    data = []
    extend_num = 0
    share_num = 0
    topic_num = 0
    for line in open(fname, encoding='utf-8'):
        tem = process_sample(line)
        if tem is None:
            continue
        url, title, abstract, tags, topic, comments = tem
        if topic is not None and topic[0] in topic_limit:
            topic_num += 1
            for comment in comments:
                real_tags = {t[0] for t in tags}
                share = len(real_tags.intersection(set(comment[0])))
                if share > 0:
                    share_num += 1
                extend = has_entity(comment[0])
                if extend:
                    extend_num += 1
                if share > 0 or extend or comment[1] > 1:  # comment[1] is voting number
                    data.append((url, title, abstract, tags, topic, comment))
    print('extend num', extend_num, 'share_num', share_num, 'topic_num', topic_num)
    json.dump(data, open('toutiao_data.json', 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    read_file('./data/toutiao.tsv')
