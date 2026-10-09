from .prepare import proposal_chunk_endpoints


def test_resume_chunks_preserve_missing_holes_and_complete_scope():
    indices=list(range(257))
    existing=set(range(100))|set(range(200,220))
    assert proposal_chunk_endpoints(indices,existing,100)==[200,257]
    assert proposal_chunk_endpoints(indices,set(indices),100)==[]
    assert proposal_chunk_endpoints(indices,set(),100)==[100,200,257]


def test_debug_prefix_does_not_expand_to_formal_sources():
    assert proposal_chunk_endpoints(list(range(257)),set(),100,limit=17)==[17]
    assert proposal_chunk_endpoints(list(range(257)),set(range(5)),100,limit=5)==[]
