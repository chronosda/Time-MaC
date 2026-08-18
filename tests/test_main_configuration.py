import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from configs.config import create_config_from_args, get_args
from configs.config import TimeMEConfig
from models.time_me import TimeMEModel
from src.TimeVLM.structured_context import StructuredContextEncoder
from utils.prompt_bank import load_dataset_prompt


class _FakeVLMManager:
    """Avoid remote model loading while testing modality dimension wiring."""

    def __init__(self, config):
        self.model = nn.Identity()
        self.hidden_size = 768
        self.vision_hidden_size = 768
        self.text_hidden_size = config.text_projection_dim


class MainConfigurationTest(unittest.TestCase):
    def test_paper_defaults(self):
        argv = ["train_time_me.py", "--data", "ETTm2", "--root_path", "unused"]
        with patch.object(sys, "argv", argv):
            config = create_config_from_args(get_args())

        self.assertEqual(config.seq_len, 512)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.batch_size, 6)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.vlm_type, "mae")
        self.assertEqual(config.context_encoder_type, "structured")
        self.assertEqual(config.dataset_embedding_dim, 32)
        self.assertEqual(config.context_hidden_dim, 128)
        self.assertEqual(config.context_output_dim, 256)
        self.assertEqual(
            config.text_encoder_name,
            "google/bert_uncased_L-2_H-128_A-2",
        )
        self.assertEqual(config.text_projection_dim, 256)
        self.assertTrue(config.use_reconstruction_mae)
        self.assertTrue(config.use_reconstruction_features)
        self.assertTrue(config.use_enhanced_fusion)
        self.assertTrue(config.mae_load_ckpt)

    def test_all_main_datasets_have_nonempty_prompts(self):
        datasets = [
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "electricity",
            "weather",
            "traffic",
        ]
        for dataset in datasets:
            with self.subTest(dataset=dataset):
                content, path = load_dataset_prompt(dataset)
                self.assertTrue(path.is_file())
                self.assertTrue(content.strip())

    def test_text_encoder_options_can_be_overridden(self):
        argv = [
            "train_time_me.py",
            "--data",
            "ETTm2",
            "--root_path",
            "unused",
            "--text_encoder_name",
            "bert-base-uncased",
            "--context_encoder_type",
            "bert",
            "--text_projection_dim",
            "768",
        ]
        with patch.object(sys, "argv", argv):
            config = create_config_from_args(get_args())

        self.assertEqual(config.text_encoder_name, "bert-base-uncased")
        self.assertEqual(config.context_encoder_type, "bert")
        self.assertEqual(config.text_projection_dim, 768)

    def test_structured_context_parameter_count_and_shape(self):
        config = TimeMEConfig()
        config.data = "ETTm2"
        encoder = StructuredContextEncoder(config)
        inputs = torch.randn(3, config.seq_len, config.enc_in)
        outputs = encoder(inputs)

        self.assertEqual(outputs.shape, (3, 256))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(encoder.get_model_info()['total_params'], 38_240)
        self.assertEqual(encoder.get_model_info()['trainable_params'], 38_240)

    @patch("models.time_me.VLMManager", _FakeVLMManager)
    def test_vision_and_text_dimensions_are_wired_separately(self):
        config = TimeMEConfig(use_gpu=False, text_projection_dim=256)
        config.device = "cpu"
        model = TimeMEModel(config)

        multimamba = model.coupled_mamba_fusion.multimamba
        self.assertEqual(
            multimamba.vision_projection.input_proj.in_features,
            768,
        )
        self.assertEqual(
            multimamba.text_projection.input_proj.in_features,
            256,
        )


if __name__ == "__main__":
    unittest.main()
