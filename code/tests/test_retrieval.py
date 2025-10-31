from retrieval import retrieve

def test_retrieval():
    results = retrieve([0]*768)
    assert isinstance(results, list)
