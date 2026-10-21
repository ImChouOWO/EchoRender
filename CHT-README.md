# EchoRender

[English](README.md)

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Architecture](https://img.shields.io/badge/Student-SSM%20%2F%20Swin%20%2F%20ViT-purple) ![License](https://img.shields.io/badge/License-MIT-green)

將凍結的 DLSS-NR Teacher 蒸餾至可訓練的 PyTorch Student。每張影像獨立處理，不使用 optical flow 或跨影格狀態。

```text
乾淨影像 -> 凍結 Teacher -> 快取 head／中間特徵／RGB 目標
同一裁切 -> 可選增強 -> Student -> 蒸餾 loss＋RGB loss
```

<details>
<summary><em>當前限制</em></summary>

Student 是研究用結構模型，不代表 NVIDIA 的數值等價實作。自訂 SSM 採用純 PyTorch 逐步掃描，並非官方 Mamba／VMamba 融合核心。評估指標衡量與 Teacher RGB 的一致性，不代表相對獨立真實標註的品質提升。目前未實作完整訓練狀態續訓或 Student 影像推論 CLI。

</details>

---

## 簡介

Teacher 使用已解包、`fully_logical=true` 的 logical safetensors，透過鎖定版本的 MLX-DLSS PyTorch 後端執行 71-block HT 結構。此訓練流程不支援 packed 或 small-topology 權重。

### 模型結構

由 [config/model.yaml](config/model.yaml) 定義。以下為目前 Student 的通道與深度，不是 Teacher block 編號。

| Stage | 模組 | 通道 | Blocks |
|---|---|---:|---:|
| E0 | Swin | 32 | 2 |
| E1 | SSM | 32 | 2 |
| E2 | SSM | 64 | 4 |
| E3 | SSM | 128 | 6 |
| E4 | Swin | 512 | 8 |
| Bottleneck | ViT | 1024 | 8 |
| Decoder E4／E3／E2／E1／E0 | Swin | 256／128／64／32／32 | 8／8／6／4／5 |

E4 將輸入的 256 通道投影至 512。decoder 第一個 Upsample 也輸出 512 通道以進行 skip 相加，下一個 Swin stage 再投影至 256。所有 `Add` 輸入形狀必須相同。

E1／E2／E3 與 bottleneck 提供蒸餾特徵。可訓練的 1×1 projection 將 Student 通道對齊 Teacher 的 64／128／256／1024；共用通道初始化為 identity，額外輸出通道初始化為零。通道不同時必須設定 `loss.projection: true`。

---

## 專案結構

```text
EchoRender/
├── README.md / CHT-README.md
├── config/model.yaml             # Student 結構
├── config/train.yaml             # 訓練策略與資料集
├── model/card/                   # SSM、attention 模組與 loss
├── model/student.py / teacher.py
├── tool/                        # 資料、快取、訓練與監測
├── docs/TEACHER_INFERENCE.md
├── weights/                     # 本機 Teacher 權重，不納入 Git
├── output/feats/                 # 可重用 Teacher 目標，不納入 Git
├── train.py / infer.py
└── pyproject.toml / uv.lock
```

---

## 安裝

### 1. 建立環境

在專案根目錄執行：

```bash
uv sync --locked
```

環境固定使用 Python 3.12。Linux／Windows 使用匹配的 CUDA 12.8 PyTorch／torchvision wheels，也支援 CPU 執行。MLX-DLSS revision 鎖定於 `pyproject.toml` 與 `uv.lock`；更換後端後須重新驗證特徵擷取。

### 2. 準備本機檔案

將解包後的 Teacher 權重放至 `weights/dlssnr-logical.safetensors`，解壓縮 MPI-Sintel-complete，並在既有訓練 YAML 設定 `dataset.root`。LPIPS 預設使用官方預訓練 AlexNet，首次初始化會下載 backbone。未另設 `TORCH_HOME` 時，訓練預設使用 `.cache/torch`。

---

## 資料集格式

```text
MPI-Sintel-complete/
├── training/clean/<scene>/frame_0001.png
├── training/final/<scene>/frame_0001.png
└── test/clean/<scene>/frame_0001.png
```

目前使用 `clean` pass：19 個訓練場景與 4 個驗證場景（`alley_1`、`alley_2`、`temple_2`、`temple_3`），目前本機資料分別為 864／200 張。訓練與驗證場景不得重疊；官方 test 資料獨立保留。`clean`／`final` 是渲染版本，不會自動視為輸入與 GT 配對。

共用中心裁切為 **192×192**。Teacher 相容前處理會補齊至最小 320 且為 64 倍數，因此目前實際網路輸入為 **320×320**。Head／RGB loss 排除 padding。目前五尺度 MS-SSIM 要求真實影像尺寸大於 160；LPIPS 至少需要 64×64。

---

## Teacher 特徵快取

| `training.releas` | 行為 |
|---|---|
| `true` | 建立或重用目標，記錄固定 `blend_scale`，完整釋放 Teacher，再由 Student 使用快取訓練。 |
| `false` | 凍結 Teacher 並保持 eval，與 Student 同時位於指定 GPU，使用即時蒸餾；不讀寫快取或顯示刪除提示。 |

`releas` 為既有設定鍵名。快取固定放在專案根目錄 `output/feats/`，包含乾淨影像的 E1／E2／E3／ViT、raw 四通道 head 與組合 RGB。依 split 與 sample ID 配對；它們是 loss 目標，不是 Student 輸入。

完整快取跨 epoch、跨次訓練重用；相容的部分快取逐張續接，只重建缺少或截斷的項目。每個檔案原子寫入後即更新索引，程序鎖保護快取至刪除提示結束。有效性包含 Teacher 權重、資料身分、裁切、前處理、capture／後端程式、裝置與 benchmark 設定。Student 結構、batch、增強或 optimizer 變更不會使乾淨目標失效。

產生快取時使用 `torch.inference_mode()`，重疊 CPU 讀圖／前處理、單一 Teacher GPU 推論與背景寫入。讀寫 worker 依首筆耗時估算，並受 CPU／cgroup／RAM 與佇列上限限制；它與 `loader.num_workers` 分開設定。

支援的 CUDA 快取擷取使用 batch-1 CUDA Graph，並預先處理 attention bias。首次 capture 比較 head 與所有擷取特徵，要求與參考路徑完全一致。不支援時顯示原因並回到參考路徑，不會默默忽略 OOM。Teacher 卸載時清除 graph 資源。Head-only benchmark 量測的是參考推論，不是 graph 快取吞吐量。

成功訓練後提示：

```text
[Done]: Delete cached features? [Y/N]:
```

`Y` 清空快取；`N`、EOF 或中斷時保留。訓練失敗也保留已完成的擷取進度。

---

## 模型與訓練設定

修改既有 [config/train.yaml](config/train.yaml)，以下僅節錄主要設定：

```yaml
model:
  student_config: model.yaml
  student_weights: null
  teacher_weights: weights/dlssnr-logical.safetensors
runtime:
  device: cuda:1
  cpu_threads: 4
  output_dir: output/test
training:
  epochs: 10
  gradient_checkpointing: false
  micro_batch_size: 4
  releas: true
  max_steps: null
  crop_size: 192
  optimizer:
    lr: 0.001
    momentum: 0.0
    weight_decay: 0.0
  grad_clip: 1.0
loader:
  batch_size: 16
  num_workers: 0
  seed: 42
  train_shuffle: true
schedule:
  kind: cosine
  eta_min: 0.000001
```

Teacher 權重相對專案根目錄解析，且必須位於其 `weights/` 下；其他路徑相對訓練 YAML。因此 `output/test` 會解析為 `config/output/test`。已存在的 run 目錄保留，新訓練依序使用 `_2`、`_3` 等後綴；選定的絕對路徑會印出並保存。

`runtime.device` 支援 `auto`、`cpu`、`cuda`、`cuda:N`。CUDA 編號對應可見裝置，設定 `CUDA_VISIBLE_DEVICES` 後可能重新編號。

### 訓練策略

`loader.batch_size` 是每次 optimizer 更新的有效張數；`training.micro_batch_size` 是同時前向的最大張數。預設為 **4 張×4 組**，以 16 張平均梯度，裁剪後更新一次。最後不足一批時使用實際張數平均；不同網路尺寸分組處理，不改變原圖幾何。

`training.gradient_checkpointing: true` 會在反向時重算各 SSM block，減少保留張量；`false` 保留張量並省去重算。驗證／推論不使用 checkpointing。

[config/model.yaml](config/model.yaml) 的 `ssm.rotation` 支援 `clockwise`／`counterclockwise`，`ssm.directions` 支援 1～4。目前雙方向順時針依序為 **右＋下 → 下＋左 → 左＋上 → 上＋右**。每個 stage 重新開始，每列／每欄狀態重設。詳見 [SSM 說明](model/card/MAMBA.md)。

SGD 預設 momentum 與 weight decay 均為零。每次 optimizer 更新前，以 cosine 調整 LR；總步數為 `epochs * len(train_loader)`，設定 `max_steps` 時取較小者。第一步使用 0.001，最後一步使用 0.000001；只有一步時維持初始 LR。不使用 warmup／restarts，loss 權重固定，快取與驗證不推進排程。設定 `kind: constant` 可停用退火。

### Student 增強

被選中的影像加入 Gaussian noise，再均勻隨機選擇亮度、對比或銳利度其中一種調整。機率由 10% 開始，在總 epochs 的 50% 達到 25%，從 75% 起關閉。10 epochs 時，第 1 個 epoch 為 10%、第 5～7 個為 25%、第 8～10 個為 0%。上限是抽樣機率，不保證每批實際張數；短訓練可能無法走完整段上升，`max_steps` 不會重縮放排程。

噪聲 sigma 取 0～0.02；亮度／對比倍率取 0.8～1.2，銳利度取 0.5～1.5。Teacher 目標保持乾淨，Student 前處理與 RGB 組合使用增強影像；驗證、benchmark 與推論不增強。增強使用獨立 RNG，由 `loader.seed` 設定；以 `augmentation.enabled: false` 關閉。

---

## Loss

定義於 [model/card/loss.py](model/card/loss.py)：

```text
E1, E2, E3 = 通道 L2 正規化特徵 MSE
ViT        = 0.5 * feature MSE + 排除對角線的 cosine token-affinity MSE
Lmid       = (E1 + E2 + E3 + ViT) / 4
Lout       = raw 四通道 head MSE
Ld         = 0.5 * Lmid + 0.5 * Lout
Lrgb       = 0.5 * RGB MSE + 0.2 * (1 - SSIM) + 0.3 * (1 - MS-SSIM)
Ltotal     = 3 * Ld + Lrgb
```

特徵對齊排除純 padding 區域，維持實際空間網格對應，詳見 [特徵擷取與對齊](tool/TEACHER_FEATURES.md)。Token affinity 以精確 row chunks 計算，不抽樣 tokens。Teacher 目標 detach，optimizer 僅包含 Student 與特徵 adapters。

兩路 RGB 使用 Teacher 固定的 learned blend cap。RGB forward 保留 FP16 捨入，只有捨入的 backward 使用 straight-through gradient；clamp／sigmoid 保留實際梯度。這是訓練近似，不代表 NVIDIA 數值等價。RGB 重建與 PSNR／SSIM／LPIPS 都以 Teacher RGB 為參考。

---

## 啟動、監測與驗證

```bash
uv run --locked python train.py --config config/train.yaml
uv run --locked python -m unittest tool.test_training -v
```

進度列顯示 epoch、loss、按影像加權的平均 loss、迭代速度與 GPU 使用率。NVML 透過 UUID 對應指定 CUDA 裝置；使用率涵蓋整張卡，無法取得時顯示 `N/A`。

每個 epoch 驗證完整 validation split，保存 PSNR／SSIM／LPIPS 的影像平均與等權場景平均。無限 PSNR 儲存為 `"Infinity"`。可選 batch-1 FP32 latency benchmark 使用暖機、同步與 eval，排除前處理、特徵擷取、組合與 backward。快取模式沿用產生快取時的 Teacher 參考耗時，每個 epoch 重新量測 Student。

| 輸出 | 內容 |
|---|---|
| `metrics.jsonl` | 每次更新的 loss、控制參數、增強數量、GPU 使用率與 epoch 評估。 |
| `last.pt` | v2 Student、特徵 adapters、optimizer、進度、模型／訓練設定與 blend scale。 |
| `summary.json` | Epoch 結果、參數量與 CUDA 配置峰值。 |
| `resolved_train.yaml` | 解析後的路徑與本次設定。 |

`max_steps` 限制更新次數，但仍執行驗證與 checkpoint 保存。`model.student_weights` 僅載入 Student 權重，以新的 optimizer／adapters 開始新 run。測試涵蓋梯度、projection 通道、掃描方向、batch、快取生命週期與排程；合成測試不代表真實資料品質。

目前 E4-512 模型以 crop 192、micro-batch 4、有效 batch 16、checkpointing 關閉，在 GPU 1（RTX 5000 Ada）的兩次短測更新中，配置峰值約 **29.14 GiB**、保留顯存約 **29.81 GiB**，每次更新約 **8～9 秒**。這不是完整 epoch benchmark，也不保證未來顯存需求。

權重、checkpoints、Teacher 快取與輸出目錄由 `.gitignore` 排除；`uv.lock` 保留版本控制以利重現。

---

## Teacher 推論

```bash
uv run --locked python infer.py --input input.png --weights weights/dlssnr-logical.safetensors --output outputs/teacher.png --device cuda:1
```

輸出 PNG 保持原圖尺寸。`--overwrite` 允許覆蓋既有檔案，`--report path.json` 保存耗時與 metadata。詳見 [Teacher 推論](docs/TEACHER_INFERENCE.md)。

---

## License

專案程式採用 [MIT](LICENSE)。外部相依套件、資料集及 Teacher 權重各依其原有條款。
