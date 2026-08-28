"""The synthetic caveat is enforced, not remembered.

These tests are the mechanism described in `src/athar/provenance.py`. The last one
walks every committed artifact, so a number that reaches `metrics/` without
declaring where it came from fails the suite rather than reaching the report.
"""

from __future__ import annotations

import json

import pytest

from athar import paths
from athar.provenance import (
    REAL_SOURCES,
    SYNTHETIC_SOURCES,
    Provenance,
    ProvenanceError,
    caveat_for,
    markdown_caveat,
    read_metric,
    write_metric,
)

SYNTHETIC_OK = {"seed": 20260829, "dgp_hash": "0" * 16}


@pytest.mark.parametrize("source", sorted(REAL_SOURCES))
def test_real_source_rejects_synthetic_flag(source):
    with pytest.raises(ProvenanceError, match="is real but synthetic=True"):
        Provenance(source=source, synthetic=True)


@pytest.mark.parametrize("source", sorted(SYNTHETIC_SOURCES))
def test_panel_derived_source_rejects_real_flag(source):
    with pytest.raises(ProvenanceError, match="synthetic=False was declared"):
        Provenance(source=source, synthetic=False, **SYNTHETIC_OK)


@pytest.mark.parametrize("missing", ["seed", "dgp_hash"])
def test_synthetic_must_name_its_configuration(missing):
    fields = {k: v for k, v in SYNTHETIC_OK.items() if k != missing}
    with pytest.raises(ProvenanceError, match=f"must declare {missing}"):
        Provenance(source="mmm", synthetic=True, **fields)


def test_unknown_source_is_rejected():
    with pytest.raises(ProvenanceError, match="unknown source"):
        Provenance(source="vibes", synthetic=False)


def test_synthetic_block_carries_the_caveat_and_real_does_not():
    synthetic = Provenance(source="panel", synthetic=True, **SYNTHETIC_OK).to_dict()
    real = Provenance(source="olist", synthetic=False, split="full").to_dict()
    assert synthetic["caveat"].startswith("SYNTHETIC")
    assert "caveat" not in real
    assert caveat_for(real) is None
    assert caveat_for(synthetic).startswith("SYNTHETIC")


def test_markdown_caveat_is_generated_not_typed():
    rendered = markdown_caveat({"source": "recovery", "synthetic": True})
    assert rendered.startswith("> **SYNTHETIC**")
    assert rendered.endswith("\n\n")
    assert markdown_caveat({"source": "criteo", "synthetic": False}) == ""


def test_artifact_without_provenance_is_rejected_on_read(tmp_path):
    (tmp_path / "naked.json").write_text(json.dumps({"roi": 2.4}))
    with pytest.raises(ProvenanceError, match="carries no provenance block"):
        read_metric("naked", tmp_path)


def test_payload_cannot_smuggle_its_own_provenance(tmp_path):
    with pytest.raises(ProvenanceError, match="must not carry its own"):
        write_metric(
            "smuggled",
            {"roi": 2.4, "provenance": {"source": "criteo", "synthetic": False}},
            Provenance(source="panel", synthetic=True, **SYNTHETIC_OK),
            tmp_path,
        )


def test_written_artifact_is_byte_identical_on_rewrite(tmp_path):
    """No timestamp, so an unchanged computation produces an unchanged file.

    This is what makes "the dashboard matches the code" checkable with `git diff`
    rather than asserted, the same property SPINE gives `reports/proof.json`.
    """
    args = ("stable", {"roi": 2.4, "channels": ["a", "b"]})
    kwargs = {"provenance": Provenance(source="criteo", synthetic=False, split="full")}
    first = write_metric(*args, **kwargs, directory=tmp_path).read_bytes()
    second = write_metric(*args, **kwargs, directory=tmp_path).read_bytes()
    assert first == second


def test_every_committed_artifact_declares_its_provenance():
    """The backstop: no number reaches `metrics/` anonymously.

    A synthetic artifact that lost its caveat, or any artifact written by hand
    rather than through `write_metric`, fails here rather than reaching a chart.
    """
    artifacts = sorted(paths.metrics_dir().glob("*.json"))
    if not artifacts:
        pytest.skip("no metrics artifacts written yet")
    for path in artifacts:
        document = json.loads(path.read_text())
        assert "provenance" in document, f"{path.name} carries no provenance block"
        block = document["provenance"]
        assert block.get("source") in REAL_SOURCES | SYNTHETIC_SOURCES, (
            f"{path.name} declares unknown source {block.get('source')!r}"
        )
        # Reconstructing the block re-runs every consistency rule on it.
        Provenance(
            source=block["source"],
            synthetic=block["synthetic"],
            split=block.get("split"),
            seed=block.get("seed"),
            dgp_hash=block.get("dgp_hash"),
        )
        if block["synthetic"]:
            assert block.get("caveat", "").startswith("SYNTHETIC"), (
                f"{path.name} is synthetic but carries no caveat"
            )
