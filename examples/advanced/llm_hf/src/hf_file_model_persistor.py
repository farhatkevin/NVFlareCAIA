import os
import time
import traceback

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class HFFileModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        exclude_vars=None,
        model=None,
        global_model_file_name="global_model",
        best_global_model_file_name="best_global_model",
        source_ckpt_file_full_name=None,
        filter_id=None,
        load_weights_only=False,
        model_name_or_path="allenai/OLMo-2-0425-1B-Instruct",  # Specify your model type here
        allow_numpy_conversion=True,
    ):
        super().__init__(
            exclude_vars=exclude_vars,
            model=model,
            global_model_file_name=global_model_file_name,
            best_global_model_file_name=best_global_model_file_name,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
            filter_id=filter_id,
            load_weights_only=load_weights_only,
            allow_numpy_conversion=allow_numpy_conversion,
        )
        self.model_name_or_path = model_name_or_path

    def save_model_file(self, save_path: str):
        """Save aggregated global model both as NVFlare .pt and HF folder.

        HF save uses save_pretrained(safe_serialization=True) to handle tied weights
        (e.g., LLaMA lm_head <-> embed_tokens) and writes config + safetensors.
        """
        agg_start = time.time()
        print(f"💾 SAVING MODEL FILE: {save_path}")

        # Get model state dict from NVFlare persistence manager
        save_dict = self.persistence_manager.to_persistence_dict()

        print(f"🔍 PERSISTENCE MANAGER DICT STRUCTURE:")
        print(f"  📦 Top-level keys: {list(save_dict.keys()) if isinstance(save_dict, dict) else 'Not a dict'}")
        if isinstance(save_dict, dict):
            for key, value in save_dict.items():
                if isinstance(value, dict):
                    print(f"  📂 '{key}' contains {len(value)} items")
                    if len(value) > 0:
                        sample_key = next(iter(value.keys()))
                        sample_val = value[sample_key]
                        print(f"    🔍 Sample: '{sample_key}' -> {type(sample_val)} {getattr(sample_val, 'shape', 'no shape')}")
                else:
                    print(f"  📄 '{key}' -> {type(value)}")

        # Always save the NVFlare/PT snapshot
        torch.save(save_dict, save_path + ".pt")
        print(f"✅ SAVED PyTorch model to {save_path}.pt")

        # Helper conversions
        def _to_tensor(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v)
            return v

        def _remap_key(k: str) -> str:
            # Normalize common prefixes to match HF expected names
            # Received keys can be "model.<hf_key>" and SFT can cause "model.model.<hf_key>"
            k = k.replace("model.base_model.model.", "model.")
            k = k.replace("base_model.model.", "model.")
            k = k.replace("model.model.", "model.")
            k = k.replace("module.", "")

            # Fix OLMo-specific lm_head key mismatch
            # Aggregated: "model.lm_head.weight" -> HF expects: "lm_head.weight"
            if k.startswith("model.lm_head"):
                k = k.replace("model.lm_head", "lm_head")
                print(f"  🔧 Key remapping: model.lm_head -> lm_head")

            return k

        def _as_tensor_state_dict(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    out[_remap_key(k)] = _to_tensor(v)
            return out

        try:
            hf_save_path = save_path + "_hf"
            print(f"🔄 Saving in HF format to {hf_save_path}")
            os.makedirs(hf_save_path, exist_ok=True)

            # Extract state_dict from NVFlare dict
            if isinstance(save_dict, dict) and "model" in save_dict:
                state_dict = save_dict["model"]
                print("found 'model' key in save_dict for state_dict")
            else:
                state_dict = save_dict
                print("did not find 'model' key; using entire save_dict as state_dict")

            tensor_sd = _as_tensor_state_dict(state_dict)

            # Prepare HF model
            hf_model = None
            try:
                print(f"🔧 Preparing HF model for saving...")
                if isinstance(getattr(self, "model", None), PreTrainedModel):
                    print("  ✅ Using existing PreTrainedModel")
                    hf_model = self.model
                else:
                    print(f"  🔧 Creating fresh model from config: {self.model_name_or_path}")
                    cfg = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                    hf_model = AutoModelForCausalLM.from_config(cfg)
                    print("  ⚠️ WARNING: Created fresh model with random weights!")
            except Exception as e:
                print(f"  ❌ Could not initialize HF model from config: {e}")

            if hf_model is not None:
                # Debug: Print tensor_sd keys for analysis
                print(f"  🔍 tensor_sd has {len(tensor_sd)} keys")
                print(f"  🔍 First 5 tensor_sd keys: {list(tensor_sd.keys())[:5]}")
                print(f"  🔍 HF model has {len(dict(hf_model.named_parameters()))} parameters")
                print(f"  🔍 First 5 HF model keys: {list(dict(hf_model.named_parameters()).keys())[:5]}")

                # Fingerprint BEFORE loading aggregated weights (fresh random model)
                def _tensor_fingerprint(tensor):
                    if hasattr(tensor, 'data'):
                        tensor = tensor.data
                    return f"{tensor.mean().item():.6f}±{tensor.std().item():.6f}"

                fresh_embed_fp = _tensor_fingerprint(hf_model.model.embed_tokens.weight)
                fresh_lm_head_fp = _tensor_fingerprint(hf_model.lm_head.weight) if hasattr(hf_model, 'lm_head') else "N/A"
                print(f"  🎲 BEFORE aggregated load - embed_tokens: {fresh_embed_fp}, lm_head: {fresh_lm_head_fp}")

                # Load aggregated weights into the model - STRICT MODE for debugging
                try:
                    missing, unexpected = hf_model.load_state_dict(tensor_sd, strict=True)
                    print(f"  ✅ Successfully loaded state_dict with strict=True")
                except Exception as strict_error:
                    print(f"  ❌ STRICT LOAD FAILED: {strict_error}")
                    print("  🔧 Falling back to strict=False...")
                    missing, unexpected = hf_model.load_state_dict(tensor_sd, strict=False)
                    print(f"  ⚠️ Missing keys ({len(missing)}): {missing[:10]}...")  # Show first 10
                    print(f"  ⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")  # Show first 10

                    if len(missing) > len(tensor_sd) * 0.5:  # If >50% keys missing
                        print("  🚨 CRITICAL: Most model weights not loaded! This will cause random performance.")
                    elif len(missing) > 0:
                        print(f"  ⚠️ WARNING: {len(missing)} parameters not loaded from aggregated weights")

                # Fingerprint AFTER loading aggregated weights
                loaded_embed_fp = _tensor_fingerprint(hf_model.model.embed_tokens.weight)
                loaded_lm_head_fp = _tensor_fingerprint(hf_model.lm_head.weight) if hasattr(hf_model, 'lm_head') else "N/A"
                print(f"  🎯 AFTER aggregated load - embed_tokens: {loaded_embed_fp}, lm_head: {loaded_lm_head_fp}")

                # Check if weights actually changed (more definitive test)
                embed_changed = fresh_embed_fp != loaded_embed_fp
                lm_head_changed = (fresh_lm_head_fp != "N/A" and fresh_lm_head_fp != loaded_lm_head_fp)

                # Also check actual tensor values for a few elements
                embed_tensor_changed = not torch.allclose(
                    hf_model.model.embed_tokens.weight[:5, :5],
                    tensor_sd.get('model.embed_tokens.weight', torch.zeros(1,1))[:5, :5],
                    atol=1e-6
                )

                print(f"  🔍 Embed fingerprint changed: {embed_changed}")
                print(f"  🔍 Embed tensor values match aggregated: {not embed_tensor_changed}")

                if not embed_changed:
                    print("  🚨 CRITICAL: embed_tokens weights UNCHANGED - aggregation not working!")
                else:
                    print("  ✅ embed_tokens weights successfully updated")

                if fresh_lm_head_fp != "N/A":
                    print(f"  🔍 LM head fingerprint changed: {lm_head_changed}")
                    if not lm_head_changed:
                        print("  🚨 CRITICAL: lm_head weights UNCHANGED - aggregation not working!")
                    else:
                        print("  ✅ lm_head weights successfully updated")

                if hasattr(hf_model, "tie_weights"):
                    try:
                        print("  🔗 Tying weights...")
                        hf_model.tie_weights()
                    except Exception as tie_error:
                        print(f"  ⚠️ Could not tie weights: {tie_error}")
                else:
                    print("  ℹ️ No tie_weights method found")

                # Save HF folder, using safe serialization (safetensors) and large shard to avoid splits
                hf_model.save_pretrained(hf_save_path, safe_serialization=True, max_shard_size="10GB")
                print(f"  ✅ Saved HF model to: {hf_save_path}")

            try:
                tok = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, trust_remote_code=True)
                tok.save_pretrained(hf_save_path)
                print("  ✅ Saved tokenizer files")
            except Exception as e:
                print(f"  ⚠️ Could not save tokenizer via AutoTokenizer: {e}")

            # List directory contents for debug
            try:
                print(f"  📁 HF directory contents: {os.listdir(hf_save_path)}")
            except Exception:
                pass

        except Exception as e:
            print(f"❌ ERROR saving model in HF format: {e}")
            print(f"Model was saved in PyTorch format only at {save_path}.pt")

        elapsed = time.time() - agg_start
        print(f"[AGGREGATION THROUGHPUT] Model save/aggregation time: {elapsed:.2f} seconds")
        # Optionally, save to a log file
        agg_log_file = save_path + "_aggregation_time.txt"
        with open(agg_log_file, "w") as f:
            f.write(f"Aggregation/save time: {elapsed:.2f} seconds\n")

    def handle_event(self, event: str, fl_ctx: FLContext):
        """Override to add debugging for events"""

        if event == EventType.START_RUN:
            print(f"🚀 Initializing persistor")
            self._initialize(fl_ctx)

        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            print(f"🏆 BEST MODEL EVENT: Saving best global model")
            # save the current model as the best model, or the global best model if available
            ml = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            if ml:
                print(f"  ✅ Got global model from context")
                self._get_persistence_manager(fl_ctx).update(ml)
            else:
                print(f"  ⚠️ No global model found in context")

            print(f"  💾 Saving to {self._best_ckpt_save_path}")
            self.save_model_file(self._best_ckpt_save_path)
            print(f"  ✅ Best model saved!")

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        """Save model with round information from meta"""
        import time

        agg_start = time.time()
        print(f"📥 SAVE MODEL called")
        self._get_persistence_manager(fl_ctx).update(ml)

        # Try to get round information from model meta
        current_round = None

        if hasattr(ml, "meta") and ml.meta:
            if "CURRENT_ROUND" in ml.meta:
                current_round = ml.meta["CURRENT_ROUND"]
            elif "current_round" in ml.meta:
                current_round = ml.meta["current_round"]

        # Add debug to see all available meta keys
        if hasattr(ml, "meta") and ml.meta:
            print(f"  🔍 Available meta keys: {list(ml.meta.keys() if ml.meta else [])}")

        if ml["meta"] is not None:
            print(f"  🔍 Model meta: {ml['meta']}")
            if "CURRENT_ROUND" in ml["meta"]:
                current_round = ml["meta"]["CURRENT_ROUND"]
            elif "current_round" in ml["meta"]:
                current_round = ml["meta"]["current_round"]
            print(f"  🔍 Found current round: {current_round}")

        # If we found round information
        if current_round is not None:
            print(f"  ℹ️ Current round: {current_round}")

            # Save both standard and round-specific checkpoints
            # self.save_model_file(self._ckpt_save_path)  # Standard
            round_path = f"{os.path.dirname(self._ckpt_save_path)}/round_{current_round}_model"
            self.save_model_file(round_path)  # Round-specific
        else:
            print(f"  ⚠️ No round information found in meta")
            self.save_model_file(self._ckpt_save_path)  # Save standard checkpoint only

        agg_end = time.time()
        elapsed = agg_end - agg_start
        print(f"[AGGREGATION THROUGHPUT] Aggregation+save time: {elapsed:.2f} seconds")
        # Optionally, save to a log file
        agg_log_file = os.path.join(os.path.dirname(self._ckpt_save_path), "aggregation_time.txt")
        with open(agg_log_file, "a") as f:
            f.write(f"Round: {current_round}, Aggregation+save time: {elapsed:.2f} seconds\n")
