# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import json
from grouper_topics import get_topics
import grouper_segments


def get_groups(segments, model1):
    community_extraction = grouper_segments.community_detection(segments, model1)
    #pims = community_extraction.get_communities()
    #pims = community_extraction.h_communities()
    pims = community_extraction.itr_communities()
    topics = get_topics(pims)

    return topics, pims
