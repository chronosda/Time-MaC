import sys
import unittest
from unittest.mock import patch

from configs.config import create_config_from_args, get_args
from utils.prompt_bank import load_dataset_prompt


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


if __name__ == "__main__":
    unittest.main()
