from persistent_kv_admission.policies import KVState, ValueAwarePolicy


def test_value_aware_policy_prefers_more_reusable_state():
    policy = ValueAwarePolicy()
    a = KVState("a", size_bytes=1024, prefix_tokens=1000, reuse_count=5, fan_out=3, age=1)
    b = KVState("b", size_bytes=1024, prefix_tokens=1000, reuse_count=1, fan_out=0, age=1)
    assert policy.score(a) > policy.score(b)
