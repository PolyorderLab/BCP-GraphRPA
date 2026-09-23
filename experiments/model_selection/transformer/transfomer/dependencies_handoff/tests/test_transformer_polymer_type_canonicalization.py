import copy
import random
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODEL_DIR))

from train_transformer_polymer_type import polymer_type_sequence_from_record


def example_record():
    return {
        "base_name": "synthetic",
        "system": {
            "components": [
                {
                    "blocks": [
                        {"label": "left", "segment": "A", "length": 0.1, "ends": ["raw_a", "raw_j"]},
                        {"label": "middle", "segment": "B", "length": 0.4, "ends": ["raw_j", "raw_k"]},
                        {"label": "arm", "segment": "A", "length": 0.2, "ends": ["raw_k", "raw_c"]},
                        {"label": "right", "segment": "B", "length": 0.3, "ends": ["raw_k", "raw_b"]},
                    ]
                }
            ]
        },
    }


def renamed_and_reordered(record):
    transformed = copy.deepcopy(record)
    blocks = transformed["system"]["components"][0]["blocks"]
    endpoint_names = sorted({endpoint for block in blocks for endpoint in block["ends"]})
    mapping = {name: f"arbitrary_{17 * i + 3}" for i, name in enumerate(reversed(endpoint_names))}
    for i, block in enumerate(blocks):
        block["label"] = f"unrelated_label_{99 - i}"
        block["ends"] = [mapping[block["ends"][1]], mapping[block["ends"][0]]]
    random.Random(9182).shuffle(blocks)
    return transformed


class StrictCanonicalizationTest(unittest.TestCase):
    def test_invariant_to_labels_endpoint_names_block_order_and_end_orientation(self):
        original = example_record()
        transformed = renamed_and_reordered(original)
        for representation in (
            "species_fraction_only",
            "no_shared_junction",
            "full_topology",
        ):
            for variant in ("canonical", "reversed", "randomized"):
                with self.subTest(representation=representation, variant=variant):
                    expected = polymer_type_sequence_from_record(
                        original,
                        variant=variant,
                        seed=202601,
                        representation=representation,
                        canonicalization="strict",
                    )
                    actual = polymer_type_sequence_from_record(
                        transformed,
                        variant=variant,
                        seed=202601,
                        representation=representation,
                        canonicalization="strict",
                    )
                    self.assertEqual(expected, actual)

    def test_legacy_mode_remains_available_for_old_checkpoints(self):
        tokens, numbers = polymer_type_sequence_from_record(
            example_record(), canonicalization="legacy"
        )
        self.assertEqual(len(tokens), len(numbers))
        self.assertEqual(tokens[0], "<CLS>")

    def test_structural_ablation_tiers_remove_the_intended_information(self):
        record = example_record()
        species_tokens, _ = polymer_type_sequence_from_record(
            record, representation="species_fraction_only", canonicalization="strict"
        )
        local_tokens, _ = polymer_type_sequence_from_record(
            record, representation="no_shared_junction", canonicalization="strict"
        )
        full_tokens, _ = polymer_type_sequence_from_record(
            record, representation="full_topology", canonicalization="strict"
        )
        self.assertNotIn("END1", species_tokens)
        self.assertNotIn("BRANCH", species_tokens)
        self.assertIn("END1", local_tokens)
        self.assertIn("BRANCH", local_tokens)
        self.assertFalse(any(token.startswith("J") for token in local_tokens))
        self.assertTrue(any(token.startswith("J") for token in full_tokens))


if __name__ == "__main__":
    unittest.main()
