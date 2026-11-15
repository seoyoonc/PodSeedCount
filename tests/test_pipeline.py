"""Regression test: counts on the annotated examples must stay close to the hand counts."""
from seedcounter.evaluate import evaluate_image, totals


def test_counts_close_to_ground_truth(gt_results, cfg):
    evals = []
    for e, labels, img in gt_results:
        assert img.error is None
        ev = evaluate_image(img, labels, e.pod_counts, cfg)
        assert len(ev["pods"]) == len(e.pod_counts), "wrong number of pods"
        for r in ev["pods"]:
            assert abs(r["error"]) <= max(4, 0.1 * r["truth"]), r
        evals.append(ev)
    t = totals(evals)
    assert t["rel_error"] < 0.05 and t["f1"] > 0.93, t


def test_bud_and_stem_ends_are_never_seeds(gt_results):
    for _, _, img in gt_results:
        for pod in img.pods:
            bud_x, stem_x = pod.bud_cut[0][0], pod.stem_cut[0][0]
            for s in pod.seeds:
                assert bud_x < s.centroid[0] < stem_x
