import datetime
import json
import pydgraph


def create_client_stub():
    return pydgraph.DgraphClientStub('localhost:9080')


def create_client(client_stub):
    return pydgraph.DgraphClient(client_stub)


def drop_all(client):
    return client.alter(pydgraph.Operation(drop_all=True))


def set_schema(client):
    schema = """
    keyword: string @index(exact) .
    name: string .
    meetingid: string @index(exact) .
    score: int .
    neighbour: uid @reverse .
    """
    return client.alter(pydgraph.Operation(schema=schema))


def write_node(client, node, meetingid):
    txn = client.txn()
    try:

        data = {
            'uid': '_:' + node,
            'keyword': node,
            'name': node,
            'meetingid': meetingid
        }

        assigned = txn.mutate(set_obj=data)

        txn.commit()

    finally:
        txn.discard()


def write_edge(client, node, meetingid):
    query = """query all($a: string, $m: string) {
        all(func: eq(keyword, $a)) @filter(eq(meetingid, $m)){
            uid
            expand(_all_)
        }
    }"""

    variables = {'$a': node[0], '$m': meetingid}
    res = client.query(query, variables=variables)
    ppl = json.loads(res.json)

    node_left = ppl['all'][0]['uid']

    query = """query all($a: string, $m: string) {
        all(func: eq(keyword, $a))  @filter(eq(meetingid, $m)){
            uid
            expand(_all_)
        }
    }"""

    variables = {'$a': node[1], '$m': meetingid}
    res = client.query(query, variables=variables)
    ppl = json.loads(res.json)

    node_right = ppl['all'][0]['uid']

    txn = client.txn()
    try:

        data = {
            'uid': node_left,
            'neighbour': [
                {
                    'uid': node_right
                }
            ]
        }

        assigned = txn.mutate(set_obj=data)

        txn.commit()

    finally:
        txn.discard()


def update_graph(graph_obj, meetingid):
    client_stub = create_client_stub()
    client = create_client(client_stub)

    # drop_all(client)
    set_schema(client)

    nodes = list(dict(graph_obj.nodes()).keys())
    edges = list(dict(graph_obj.edges()).keys())

    for node in nodes:
        write_node(client, node, meetingid)

    for edge in edges:
        write_edge(client, edge, meetingid)

    client_stub.close()
