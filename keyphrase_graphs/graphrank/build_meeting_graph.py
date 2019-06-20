import networkx as nx
import json as js
from .graphrank import GraphRank


def read_json(json_file):
    with open(json_file) as f:
        meeting = js.load(f)

    return meeting


def get_context_id(meeting):
    context_id = meeting['contextId']
    attrs = {'name': 'context_id'}

    return (context_id, attrs)


def get_instance_id(meeting):
    context_instance_id = meeting['contextInstanceId']
    attrs = {'name': 'context_instance_id'}

    return (context_instance_id, attrs)


def get_segment_list(meeting):
    segment_list = meeting['segments']

    return segment_list


def get_segment_nodes(segment_list):
    segment_nodes_list = []
    for segment in segment_list:
        segment_nodes = {'segment_id': segment['id'],
                         'spoken_by': segment['spokenBy'],
                         'transcribed_by': segment['transcriber']
                         }
        segment_nodes_list.append(segment_nodes)

    return segment_nodes_list


def get_segment_attrs(segment_list):
    segment_attrs_list = []
    for segment in segment_list:
        segment_edge_attrs = {'text': segment['originalText'],
                              'confidence': segment['confidence'],
                              'start_time': segment['startTime'],
                              'end_time': segment['endTime'],
                              'duration': segment['duration'],
                              'language': segment['languageCode']}

        segment_attrs_list.append(segment_edge_attrs)

    return segment_attrs_list


def construct_context_edge(meeting):
    context_id, _ = get_context_id(meeting)
    meeting_id, _ = get_instance_id(meeting)

    attrs = {'rel': 'has_meeting'}
    context_meeting_edge = (context_id, meeting_id, attrs)

    return context_meeting_edge


def construct_meeting_segment_edge(meeting):
    meeting_id, _ = get_instance_id(meeting)
    segment_list = get_segment_list(meeting)

    segment_nodes_list = get_segment_nodes(segment_list)
    attrs = {'rel': 'has_segment'}

    segment_attrs_list = get_segment_attrs(segment_list)
    meeting_segment_edge_list = []

    for i, node_val in enumerate(segment_nodes_list):
        segment_id = node_val['segment_id']
        segment_attrs = segment_attrs_list[i]
        segment_attrs.update(attrs)

        meeting_segment_edge = (meeting_id, segment_id, segment_attrs)
        meeting_segment_edge_list.append(meeting_segment_edge)

    return meeting_segment_edge_list


def construct_segment_nodes(meeting):
    segment_list = get_segment_list(meeting)

    segment_nodes_list = get_segment_nodes(segment_list)
    id_user_edge_list = []
    id_transcriber_edge_list = []

    attrs = {
        'id_user': {'rel': 'spoken_by'},
        'id_transcriber': {'rel': 'transcribed_by'}
    }
    for node_val in segment_nodes_list:
        segment_id = node_val['segment_id']
        segment_user = node_val['spoken_by']
        segment_transcriber = node_val['transcribed_by']

        id_user_edge = (segment_id, segment_user, attrs['id_user'])
        id_transcriber_edge = (segment_id, segment_transcriber, attrs['id_transcriber'])

        id_user_edge_list.append(id_user_edge)
        id_transcriber_edge_list.append(id_transcriber_edge)

    return id_user_edge_list, id_transcriber_edge_list


def build_meeting_graph(json_file, word_graph_obj=None):
    mg = nx.DiGraph(type='meeting_graph')

    meeting_data = read_json(json_file)

    context_info = get_context_id(meeting_data)
    meeting_info = get_instance_id(meeting_data)

    context_edges = construct_context_edge(meeting_data)
    meeting_segment_edge_list = construct_meeting_segment_edge(meeting_data)

    id_user_edges, id_transcriber_edges = construct_segment_nodes(meeting_data)

    # Add Nodes in the graph
    mg.add_nodes_from([context_info, meeting_info])

    # Add Word graph object as a node
    meeting_id, meeting_inst_attr = get_instance_id(meeting_data)
    mg.add_edge(meeting_id, word_graph_obj, rel='keywords')

    # Add Edges
    mg.add_edges_from([context_edges])
    mg.add_edges_from(meeting_segment_edge_list)
    mg.add_edges_from(id_user_edges)
    mg.add_edges_from(id_transcriber_edges)

    return mg
