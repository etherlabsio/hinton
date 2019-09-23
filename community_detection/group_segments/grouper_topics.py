

def get_topics(pims):
    topics = {}
    topics['topics'] = []

    for i in pims:
        # if (len(pims[i]))>=2:
        new_topic = {}
        new_topic['id'] = pims[i]['segment0'][3]
        # new_topic['text']=pims[i]['segment0'][0]
        new_topic['no_of_segment'] = len(pims[i])
        new_topic['authors'] = pims[i]['segment0'][2]
        new_topic['authoredAt'] = pims[i]['segment0'][1]
        topics['topics'].append(new_topic)

    return topics
