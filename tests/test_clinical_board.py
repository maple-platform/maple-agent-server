import unittest
import os
import sys
import types
from unittest.mock import AsyncMock, patch

# Production RAG initializes a CUDA sentence-transformer at import time. Board unit
# tests replace only that external boundary so they can run on CPU-only CI.
fake_rag_service = types.ModuleType("services.rag_service")
fake_rag_service.retrieve = AsyncMock(return_value=([], [], ""))
sys.modules["services.rag_service"] = fake_rag_service

fake_retriever = types.ModuleType("rag.retriever")
fake_retriever.search_models = lambda *_args, **_kwargs: []
sys.modules["rag.retriever"] = fake_retriever

fake_embedder = types.ModuleType("rag.embedder")
sys.modules["rag.embedder"] = fake_embedder

from routers.agent import InterpretResponse
from services import agent_service, board_service


class ConfidenceTests(unittest.TestCase):
    def test_complementary_tasks_are_not_compared(self):
        steps = [
            {
                "model": "ChestXray14",
                "task_type": "classification",
                "result_type": "classification_probabilities",
                "predictions": [{"finding": "Pneumonia", "prob": 0.61}],
            },
            {
                "model": "RSNA_YOLO",
                "task_type": "detection",
                "result_type": "bbox_overlay,detection_predictions",
                "predictions": [{"pred_name": "pneumonia_opacity", "conf": 0.87}],
            },
        ]

        result = agent_service._build_confidence(
            [{"label": "Pneumonia", "text": "폐렴 의심"}], steps
        )

        self.assertEqual(result["display"], 0.61)
        self.assertEqual(result["source_model"], "ChestXray14")
        self.assertIsNone(result["score_gap"])
        self.assertFalse(result["conflict"])
        self.assertEqual(len(result["model_scores"]), 2)

    def test_same_task_gap_is_exposed_as_conflict(self):
        steps = [
            {
                "model": "ChestXray14",
                "task_type": "classification",
                "predictions": [{"finding": "Pneumonia", "prob": 0.87}],
            },
            {
                "model": "CheXpert",
                "task_type": "classification",
                "predictions": [{"label": "Pneumonia", "score": 0.61}],
            },
        ]

        result = agent_service._build_confidence(
            [{"label": "Pneumonia", "text": "폐렴 의심"}], steps
        )

        self.assertAlmostEqual(result["score_gap"], 0.26)
        self.assertTrue(result["conflict"])
        self.assertIsNone(result["display"])
        calibration = agent_service._calibrate(
            {"claims": [{"label": "Pneumonia"}]},
            {"differential": [{"label": "Pneumonia"}]},
            {"veto": False, "risk_tier": "moderate"},
            result,
        )
        self.assertEqual(
            agent_service._decide_escalation(calibration),
            ("pending_review", "model_disagreement"),
        )

    def test_high_risk_requires_review(self):
        calibration = {
            "veto": False,
            "risk_tier": "high",
            "agreement_score": 1.0,
            "score_gap": None,
        }
        self.assertEqual(
            agent_service._decide_escalation(calibration),
            ("pending_review", "high_risk"),
        )


class BoardParsingTests(unittest.TestCase):
    def test_invalid_json_fails_closed(self):
        with self.assertRaises(ValueError):
            board_service._parse_json("not-json")


class InterpretFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_board_mode_off_uses_single_interpret_call(self):
        legacy_text = (
            "## Summary\nA preliminary finding is present. Clinical correlation is required.\n\n"
            "## Key Imaging Findings\nThe model output identifies a focal region. "
            "The full study should be reviewed.\n\n"
            "## Clinical Interpretation\nThis is supportive rather than diagnostic. "
            "Uncertainty remains in the automated result.\n\n"
            "## Recommendations or Limitations\nSpecialist review is recommended. "
            "Do not use this output as a standalone diagnosis."
        )
        steps = [{
            "step": 1,
            "model": "Model",
            "task_type": "classification",
            "result_type": "classification",
            "predictions": [],
            "model_output": {},
            "images": [],
        }]
        with (
            patch.dict(os.environ, {"BOARD_MODE": "off"}),
            patch.object(agent_service.llm_client, "generate", AsyncMock(return_value=legacy_text)) as generate,
            patch.object(board_service, "_run_reader", AsyncMock()) as reader,
            patch.object(agent_service.wiki_service, "resolve_model", return_value=None),
        ):
            response = await agent_service.interpret(
                "판독",
                {"department": "", "project": ""},
                {"mode": "prediction", "plan": {}, "attachments": []},
                steps,
            )

        self.assertEqual(response["interpretation"], legacy_text)
        self.assertEqual(response["status"], "confirmed")
        self.assertEqual(response["board"], {})
        generate.assert_awaited_once()
        reader.assert_not_awaited()
        InterpretResponse.model_validate(response)

    async def test_structured_result_and_legacy_contract(self):
        reader = {
            "finding": "흉부 영상에서 우하엽 폐렴 의심 혼탁이 관찰됩니다. [IMG:gradcam_overlay]",
            "interpretation": "분류 확률과 영상 분포를 함께 고려할 때 폐렴 가능성이 있으나 확정 진단은 아닙니다.",
            "recommendation": "임상 증상 및 검사실 소견과 연계하고 필요하면 추적 흉부 영상을 권고합니다.",
            "claims": [{"label": "Pneumonia", "text": "폐렴 의심 소견"}],
            "sentence_map": [],
        }
        challenger = {
            "differential": [{"label": "Atelectasis", "rationale": "감별이 필요합니다."}],
            "source": "blind_same_model",
        }
        evidence = {"evidence_map": [], "unsupported_claims": []}
        guardian = {
            "veto": False,
            "risk_tier": "moderate",
            "flags": [],
            "rationale": "즉시 위험 소견은 확인되지 않았습니다.",
            "evidence_refs": [],
        }
        steps = [{
            "step": "s1",
            "model": "ChestXray14",
            "task_type": "classification",
            "result_type": "classification_probabilities",
            "predictions": [{"finding": "Pneumonia", "prob": 0.61}],
            "model_output": {},
            "images": [],
        }]

        with (
            patch.dict(os.environ, {"BOARD_MODE": "on"}),
            patch.object(board_service, "_run_reader", AsyncMock(return_value=reader)),
            patch.object(board_service, "_run_challenger", AsyncMock(return_value=challenger)),
            patch.object(board_service, "_run_evidence", AsyncMock(return_value=evidence)),
            patch.object(board_service, "_run_guardian", AsyncMock(return_value=guardian)),
            patch.object(agent_service.wiki_service, "resolve_model", return_value=None),
        ):
            response = await agent_service.interpret(
                "폐렴인지 확인",
                {"department": "Pulmonology", "project": "ChestXray14"},
                {"mode": "general", "plan": {}, "attachments": []},
                steps,
            )

        parsed = InterpretResponse.model_validate(response)
        self.assertEqual(parsed.status, "confirmed")
        self.assertEqual(parsed.result.confidence.display, 0.61)
        self.assertIn("## 소견", parsed.interpretation)
        self.assertIn("감별진단 고려", parsed.result.interpretation)
        self.assertNotIn("triage", parsed.board.model_dump())

    async def test_reader_failure_requests_human_review(self):
        steps = [{
            "step": 1,
            "model": "Model",
            "task_type": "classification",
            "result_type": "classification",
            "predictions": [],
            "model_output": {},
            "images": [],
        }]
        with (
            patch.dict(os.environ, {"BOARD_MODE": "on"}),
            patch.object(board_service, "_run_reader", AsyncMock(side_effect=ValueError("bad json"))),
            patch.object(agent_service.wiki_service, "resolve_model", return_value=None),
        ):
            response = await agent_service.interpret(
                "판독",
                {"department": "", "project": ""},
                {"mode": "prediction", "plan": {}, "attachments": []},
                steps,
            )

        self.assertEqual(response["status"], "pending_review")
        self.assertEqual(response["escalation_reason"], "board_component_failure")
        InterpretResponse.model_validate(response)


if __name__ == "__main__":
    unittest.main()
