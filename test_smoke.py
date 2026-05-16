from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoclimate_fusionllm.models import MMWSTM_ADRAN_Plus
from geoclimate_fusionllm.utils import count_parameters


def test_model_instantiates():
    model = MMWSTM_ADRAN_Plus(input_dim=30, hidden_dim=32)
    assert count_parameters(model) > 0
