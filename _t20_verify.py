import torch

ck = torch.load("../checkpoints/checkpoint_epoch_0095.pth", map_location="cpu", weights_only=False)
print("epoch:", ck["epoch"])
print("val_loss:", ck["val_loss"])
print("best_val_loss:", ck.get("best_val_loss"))
print("stage:", ck.get("current_stage"))
print("has_model:", "model_state_dict" in ck, "| has_residual:", ck.get("residual_state_dict") is not None)

# 同时核对当前(被覆盖的) best_model.pth 现状，供 INC-0003 取证
bm = torch.load("../checkpoints/best_model.pth", map_location="cpu", weights_only=False)
print("current best_model.pth: epoch", bm["epoch"], "val_loss", bm["val_loss"])