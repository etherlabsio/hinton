import json
from grouper_topics import get_topics
import grouper_segments


def get_groups(segments, model1):
    community_extraction = grouper_segments.community_detection(segments, model1)
    #pims = community_extraction.get_communities()
    pims = community_extraction.h_communities()
    topics = get_topics(pims)

    return topics, pims
