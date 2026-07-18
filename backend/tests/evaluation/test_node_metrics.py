def test_node_metrics_report_precision_recall_and_f1() -> None:
    from moonlightbox.evaluation.node_metrics import NodeLabel, evaluate_nodes

    gold = [
        NodeLabel("conflict", frozenset({"m1", "m2"})),
        NodeLabel("reconciliation", frozenset({"m9", "m10"})),
    ]
    predicted = [
        NodeLabel("conflict", frozenset({"m1", "m2"})),
        NodeLabel("cold_war", frozenset({"m5"})),
    ]

    metrics = evaluate_nodes(predicted, gold)

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
