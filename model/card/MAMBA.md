# Pure PyTorch Mamba / SSM

## 技術標籤

`PyTorch` `Selective SSM` `Gradient Checkpointing`

## 簡介

`mamba.py` 提供純 PyTorch 的 Mamba-1 style selective SSM。這是自訂研究實作，
不等同官方 Mamba／VMamba，也不宣稱具有其融合核心的效能。

`Mamba` 為 BLC mixer，`MambaBlock`／`MambaStage` 為 BCHW block／stage；
`SSMStage` 是已註冊的別名。各方向共用 mixer 權重，每列／每欄的狀態從零開始，
影像與批次成員之間不共享狀態。

在 `config/model.yaml` 的 `ssm` 區塊統一定義 E1／E2／E3 的輪轉方向與每個 block 的掃描方向數，
各方向輸出取平均後接回 residual。目前設定為順時針、每次兩個方向。
各 stage 從第一組開始，每四個 blocks 重複：

| Block | 方向 |
|---|---|
| 1 | 左→右 ＋ 上→下 |
| 2 | 上→下 ＋ 右→左 |
| 3 | 右→左 ＋ 下→上 |
| 4 | 下→上 ＋ 左→右 |

`rotation` 支援 `clockwise`／`counterclockwise`，`directions` 支援 1～4。
從向右開始，每個 block 沿輪轉方向前進一格，再選取連續的指定數量方向。
例如單方向順時針依序為右、下、左、上；四方向時每個 block 都涵蓋四個方向。
當 `training.gradient_checkpointing: true` 且訓練啟用梯度時，
每個 SSM block 使用 `checkpoint(..., use_reentrant=False)`，
反向時重算中間狀態以降低顯存。設為 `false` 則保留中間張量、不重算；目前預設為 false。
eval／no_grad／inference_mode 不使用 checkpoint。

核心包含輸入相關 delta/B/C、負對角 A、causal depthwise convolution、D skip 與 SiLU gate。
半精度輸入的 recurrence 仍以 FP32 計算；FP64 保留供梯度檢查。
不要整體重新初始化 `dt_proj.bias`；`_no_weight_decay` 僅為提示，optimizer 必須主動分組才會生效。

## 環境建置

使用專案鎖定的 PyTorch 環境，無需 mamba-ssm、Triton 或額外 CUDA extension。
模型 YAML 的 SSM stage 參數依序為 channels、depth、state_dim、d_conv、expand。
輪轉方式只需在頂層設定一次：

```yaml
ssm:
  rotation: clockwise
  directions: 2
backbone:
  - [-1, 1, SSMStage, [64, 4, 16, 4, 2]]
```

訓練設定中 `loader.batch_size` 是每次更新的有效張數；`training.micro_batch_size`
是同時前向的最大張數。梯度以實際有效張數平均，包含最後不足一組的情況。

## 參考資料

- [Mamba 官方實作](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)

## License

依專案既有 LICENSE；不包含 Teacher 權重的額外授權。
