from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PreTrainedModel
import torch
import os
import time
import traceback
import numpy as np

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.model import ModelLearnable

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
        model_name_or_path="meta-llama/llama-3.2-1b",  # Specify your model type here
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
                print(f"using autoconfig for saving model ")
                if isinstance(getattr(self, "model", None), PreTrainedModel):
                    hf_model = self.model
                else:
                    cfg = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                    hf_model = AutoModelForCausalLM.from_config(cfg)
            except Exception as e:
                print(f"  ⚠️ Could not initialize HF model from config: {e}")

            if hf_model is not None:
                # Load aggregated weight s into the model (tolerate missing/unexpected)
                missing, unexpected = hf_model.load_state_dict(tensor_sd, strict=False)
                if hasattr(hf_model, "tie_weights"):
                    try:
                        print("tie weights")
                        hf_model.tie_weights()
                    except Exception:
                        print("  ⚠️ Could not tie weights")
                        pass
                else:
                    print("no tie weights method found")

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
