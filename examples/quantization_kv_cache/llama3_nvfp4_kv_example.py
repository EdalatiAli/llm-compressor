from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure NVFP4 KV cache quantization with separate FP8 query calibration.
#
# kv_cache_scheme: NVFP4 two-level scaling for K and V tensors:
#   - Global scale (FP32): calibrated per-tensor over the dataset (static)
#   - Per-group scale (FP8): computed dynamically at runtime (group_size=16)
#   dynamic=DynamicType.LOCAL tells the calibrator to only compute the global
#   scale statically; the per-group FP8 scales are left for the runtime.
#
# q_scheme: per-tensor FP8 calibration for Q tensors, independent of K/V.
#   This allows Q to use simpler per-tensor scaling while K/V use NVFP4.
nvfp4_kv_args = QuantizationArgs(
    num_bits=4,
    type="float",
    strategy=QuantizationStrategy.TENSOR_GROUP,
    symmetric=True,
    dynamic=DynamicType.LOCAL,
    group_size=16,
    observer="static_minmax",
    scale_dtype=FP8_E4M3_DATA.dtype,
    zp_dtype=FP8_E4M3_DATA.dtype,
)

fp8_q_args = QuantizationArgs(
    num_bits=8,
    type="float",
    strategy=QuantizationStrategy.TENSOR,
    symmetric=True,
    dynamic=False,
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=nvfp4_kv_args,
    q_scheme=fp8_q_args,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-nvfp4-kv"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
