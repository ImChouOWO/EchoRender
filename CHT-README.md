# EchoRender

[English](README.md)

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Architecture](https://img.shields.io/badge/Student-SSM%20%2F%20Swin%20%2F%20ViT-purple) ![License](https://img.shields.io/badge/License-MIT-green)

將凍結的 DLSS-NR Teacher 蒸餾至可訓練的 PyTorch Student。以 Sintel clean 的 1～10 幀連續片段訓練，使用 Sintel 光流重投影歷史，並由同一 Student 支援四種 profile。

```text
乾淨影像 -> 凍結 Teacher -> 快取 head／中間特徵／RGB 目標
同一 RGB 裁切 -> 可選增強 -> RGB-only Student -> 蒸餾 loss＋RGB loss
```

<details>
<summary><em>當前限制</em></summary>

Student 是研究用結構模型，不代表 NVIDIA 的數值等價實作。自訂 SSM 採用純 PyTorch 逐步掃描，並非官方 Mamba／VMamba 融合核心。評估指標衡量與 Teacher RGB 的一致性，不代表相對獨立真實標註的品質提升。目前未實作完整訓練狀態續訓；`val.py` 提供既有權重驗證與比較圖輸出。

</details>

---

## 簡介

Teacher 使用已解包、`fully_logical=true` 的 logical safetensors，透過鎖定版本的 MLX-DLSS PyTorch 後端執行 71-block HT 結構。此訓練流程不支援 packed 或 small-topology 權重。

### 模型結構

由 [config/model.yaml](config/model.yaml) 定義。Student 以 `input_channels: 3` 接收 `[0,1]` 的原始 RGB，影像通道不加入模型噪聲或控制資料；另外接收控制向量，並在內部融合自身前一幀的 RGB 歷史。Teacher 維持 16 通道前處理、四通道 head 與 E1／E2／E3／ViT 蒸餾目標；兩路使用相同的鏡像 padding。目前 checkpoint 格式為 `echorender-rgb-temporal-v3`；舊 Student 權重須重新訓練，舊單張 Teacher 快取不會沿用。以下為目前 Student 的通道與深度，不是 Teacher block 編號。

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
├── train.py / val.py / infer.py
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

`releas` 為既有設定鍵名。快取固定放在專案根目錄 `output/feats/`，包含乾淨影像的 E1／E2／E3／ViT、raw 四通道 head 與組合 RGB。以 profile、split、序列與幀識別配對，路徑為 `output/feats/<profile>/<split>/<source>/<scene>/<frame>/targets.pt`；同檔保存 head、中間特徵、RGB 目標、歷史及識別資料。它們是 loss 目標，不會拿 Teacher 歷史餵給 Student。

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

Teacher 權重與 `runtime.output_dir` 相對專案根目錄解析；權重必須位於其 `weights/` 下。`output/test` 會解析為 `EchoRender/output/test`，其他路徑仍相對訓練 YAML。已存在的 run 目錄保留，新訓練依序使用 `_2`、`_3` 等後綴；選定的絕對路徑會印出並保存。

`runtime.device` 支援 `auto`、`cpu`、`cuda`、`cuda:N`。CUDA 編號對應可見裝置，設定 `CUDA_VISIBLE_DEVICES` 後可能重新編號。

### 訓練策略

`loader.batch_size` 是每次更新的有效片段數，每個片段對應一種 profile；`training.micro_batch_size` 限制同時前向的幀數。先平均各片段內的逐幀 loss，再依有效 batch 平均梯度。每一幀之間切斷歷史梯度，避免反向圖隨幀數增加；片段越長仍需要更多運算。`rendering.train_min_frames` 到 `max_frames` 控制隨機前綴長度，上限 10 幀。

`training.gradient_checkpointing: true` 會在反向時重算各 SSM block，減少保留張量；`false` 保留張量並省去重算。驗證／推論不使用 checkpointing。

[config/model.yaml](config/model.yaml) 的 `ssm.rotation` 支援 `clockwise`／`counterclockwise`，`ssm.directions` 支援 1～4。目前雙方向順時針依序為 **右＋下 → 下＋左 → 左＋上 → 上＋右**。每個 stage 重新開始，每列／每欄狀態重設。詳見 [SSM 說明](model/card/MAMBA.md)。

SGD 預設 momentum 與 weight decay 均為零。每次 optimizer 更新前，以 cosine 調整 LR；總步數為 `epochs * len(train_loader)`，設定 `max_steps` 時取較小者。第一步使用 0.001，最後一步使用 0.000001；只有一步時維持初始 LR。不使用 warmup／restarts，loss 權重固定，快取與驗證不推進排程。設定 `kind: constant` 可停用退火。

### Student 增強

被選中片段內的每張訓練影像，先做相同的一種亮度／對比／銳利度調整，再加入 Gaussian 噪聲。
排程比例同時控制增強樣本上限與強度：`factor = 1 + ratio`，`sigma = noise_std * ratio`。
預設倍率為 1.10～1.25，sigma 為 0.02～0.05；總 epoch 的 50% 到達最大比例 25%，
從 75% 起關閉。10 epochs 時，第 1 個為 10%，第 5～7 個為 25%，第 8～10 個為 0%。
增強數向下取整，並同時受片段與幀數比例上限限制；其餘保留原圖。`max_steps` 不重算 epoch 排程。

```yaml
augmentation:
  enabled: true
  start_ratio: 0.10
  max_ratio: 0.25
  peak_fraction: 0.50
  stop_fraction: 0.75
  noise_std: 0.20
```

Teacher 目標保持乾淨且凍結；Student 前處理與 RGB 合成使用增強影像。
原圖驗證評分、特徵快取重用與 benchmark 維持既有流程。
增強使用獨立 RNG，由 `loader.seed` 設定；以 `augmentation.enabled: false` 關閉訓練增強。
驗證比較圖每次只使用一種模式，強度由報表設定控制。

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
uv run --locked python -m unittest tool.test_training tool.test_reporting tool.test_val -v
```

進度列顯示 epoch、loss、按影像加權的平均 loss、迭代速度與 GPU 使用率。NVML 透過 UUID 對應指定 CUDA 裝置；使用率涵蓋整張卡，無法取得時顯示 `N/A`。

每個 epoch 驗證完整 validation split，保存 PSNR／SSIM／LPIPS 的影像平均與等權場景平均。無限 PSNR 儲存為 `"Infinity"`。可選 batch-1 FP32 latency benchmark 使用暖機、同步與 eval，排除前處理、特徵擷取、組合與 backward。快取模式沿用產生快取時的 Teacher 參考耗時，每個 epoch 重新量測 Student。

| 輸出 | 內容 |
|---|---|
| `metrics.jsonl` | 每次更新的 loss、控制參數、增強數量、GPU 使用率與 epoch 評估。 |
| `best.pt` | 依驗證指標選出的最佳 checkpoint，內容格式與 `last.pt` 相同。 |
| `last.pt` | v3 條件化 RGB Student、特徵 adapters、optimizer、進度、模型／訓練設定與 blend scale。 |
| `summary.json` | Epoch 結果、參數量與 CUDA 配置峰值。 |
| `resolved_train.yaml` | 解析後的路徑與本次設定。 |

`max_steps` 限制更新次數，但仍執行驗證與 checkpoint 保存。`model.student_weights` 僅載入 Student 權重，以新的 optimizer／adapters 開始新 run。測試涵蓋梯度、projection 通道、掃描方向、batch、快取生命週期與排程；合成測試不代表真實資料品質。

改為 RGB 輸入前的 16 通道 E4-512 Student 以 crop 192、micro-batch 4、有效 batch 16、checkpointing 關閉，在 GPU 1（RTX 5000 Ada）的兩次短測更新中，配置峰值約 **29.14 GiB**、保留顯存約 **29.81 GiB**，每次更新約 **8～9 秒**。這不是完整 epoch benchmark，也不保證未來顯存需求。

權重、checkpoints、Teacher 快取與輸出目錄由 `.gitignore` 排除；`uv.lock` 保留版本控制以利重現。

---

## Teacher 推論

```bash
uv run --locked python infer.py --input input.png --weights weights/dlssnr-logical.safetensors --output output/teacher.png --device cuda:1
```

輸出 PNG 保持原圖尺寸。`--overwrite` 允許覆蓋既有檔案，`--report path.json` 保存耗時與 metadata。詳見 [Teacher 推論](docs/TEACHER_INFERENCE.md)。

---

## 最佳權重與最終報表

訓練只保存 `best.pt` 與 `last.pt`。預設依驗證集影像平均 PSNR 最高選擇 best；
`reporting.best_metric` 也支援 `loss`、`ssim`、`lpips`，其中 loss／LPIPS 取最小，
PSNR／SSIM 取最大，相同分數保留原 best。更新時印出
`[Best]: save best weight | ...`，列出相對前一個 best 的各指標改善百分比；
正值代表改善，無法定義的百分比顯示 `N/A`。

結束時依各 epoch 的平均值列印 train loss、val loss、PSNR、SSIM、LPIPS 的
max／min／mean／best。每個指標各自的最佳值，與所選 best checkpoint 當時的值
會分開記錄。報表輸出至 `runtime.output_dir` 同層的 `val/<project_name>/`，預設為專案根目錄的 `output/val/`；
null 沿用本次訓練 run 名稱，遇到既有報表目錄會增加流水號。

```yaml
reporting:
  enabled: true
  project_name: null
  best_metric: psnr
  crop_size: 0
  inference_size: 320
  samples_per_mode: 2
  strengths: [0.25, 0.50, 0.75, 1.00]
```

`statistics.json`、`statistics.csv`、`epochs.csv` 與 `training_metrics.png`
保存統計與四格曲線。訓練進度採固定欄位顯示 epoch、GPU 顯存／使用率、loss、
LR 與影像數；驗證另顯示 PSNR、SSIM、LPIPS。

統計後依序載入最佳 Student 與凍結 Teacher，推論驗證與測試影像。
各 split 對 original、noise、brightness、contrast、sharpness 各隨機抽 2 張，
資料足夠時共 10 張來源影像。`loader.seed` 固定抽樣，兩個模型共用相同來源與輸入。
小型 split 可跨模式重用來源，每種模式最多選 `samples_per_mode` 張不同影像。
原圖驗證評分仍處理完整 split。

驗證每次只套用一種增強。四級 noise sigma 為 0.05／0.10／0.15／0.20；
亮度、對比、銳利度的倍率為 1.25／1.50／1.75／2.00。
噪聲使用同一圖樣依強度縮放。非 original 模式依序排列原圖與四級增強；
original 只有一欄。每個檔案只對應一張來源影像，檔名保留影像身分。

```text
output/val/<project_name>/
├── statistics.json / statistics.csv / epochs.csv / training_metrics.png
├── predictions.json / predictions.jsonl / predictions.csv
├── preview/<mode>/<split>__<sample_id>.png
└── pred/
    ├── student/<mode>/<split>__<sample_id>.png
    └── teacher/<mode>/<split>__<sample_id>.png
```

preview 為一列輸入；pred 第一列為各強度輸入，第二列為對應預測。
粗體英文標題包含 `| total infer 1.234 s | avg infer 0.247 s |`。
時間單位為秒，GPU 會前後同步；計算純模型 forward，排除每種輸入形狀的首次暖機、
前處理、RGB 合成與存檔。總時間與平均時間均以該圖實際顯示的欄位計算。
各欄維持長寬比，顯示寬度最多 384 像素。

`reporting.crop_size: 0` 保留完整來源。`reporting.inference_size: 320` 限制推論長邊，
不放大小圖；增強在 resize 後、特徵前處理前執行，避免縮圖稀釋噪聲。
相同的輸入增強差值會映射回原始尺寸供顯示。推論尺寸先完成 RGB 合成，
再將 RGB 預測差值映射至原始尺寸的增強輸入。inference_size 設為 0 可使用原尺寸推論，
正值 crop_size 則在此流程前先中心裁切。

推論索引記錄抽樣名單、增強參數、排列尺寸、輸入／特徵／預測雜湊及時間。
`predictions.csv` 保存各版本推論秒數，以及相對同一來源原圖預測的 RGB RMSE。
缺少測試資料時記錄 `completed_with_missing_splits`，仍輸出可用影像。
失敗時保留特徵快取，報表完成後才詢問是否刪除。
`reporting.enabled: false` 跳過最終圖表與圖片，仍保存 best／last 及列印統計。

此功能套用至新啟動的訓練程序。舊訓練若只保留 `last.pt`，無法還原較早 epoch 的最佳權重。

---

## 獨立驗證與預測圖輸出

`val.py` 與 `train.py` 位於同級，可直接驗證既有 checkpoint 並重新輸出比較圖：

```bash
uv run --locked python val.py --weights output/test_12/last.pt --project my_validation --imgsz 320
# 只選驗證集一張影像試跑：
uv run --locked python val.py --weights output/test_12/last.pt --project preview --split val --max-clips 1
```

以 `best.pt` 載入選定的最佳模型，或以 `last.pt` 載入最後一個模型。
模型結構、特徵 adapters、驗證 crop 與 loss 取自 checkpoint；
資料路徑、裝置、增強與比較圖設定取自 `--config config/train.yaml`。
可用 `--device cuda:1` 覆蓋設定的裝置。

`--project` 定義報表名稱。`runtime.output_dir: output/test` 時，報表輸出至
`output/val/<project>/`；重複名稱自動加上 `_2`、`_3`，不覆蓋既有結果。
`--split` 支援 `val`、`test`、`both`（預設）；`--max-clips N` 限制各 split 的
實體片段數，並保留四種 profile 的評分與圖片輸出。單獨指定不存在的 split 會報錯；`both` 則記錄缺失並處理可用資料。

`validation.json` 保存重新計算的乾淨影像 loss／PSNR／SSIM／LPIPS（Teacher RGB 參考）
與所選權重 epoch。若 checkpoint 同目錄有 `summary.json`，會額外重建訓練歷史曲線
及統計，其最佳 epoch 與手動選取的 `last.pt` 分開記錄。
缺少歷史資料時仍可驗證與輸出圖片，不會產生虛構曲線。
驗證使用 checkpoint 的訓練 crop；比較圖由 `reporting.crop_size` 控制裁切，
由 `reporting.inference_size` 或 `--imgsz 320` 控制縮放推論尺寸；
獨立驗證使用同一快取根目錄下的 `evaluate` 分類與 `evaluation.json` 索引，不覆寫訓練目標；Teacher 先完成擷取並卸載，才開始 Student 評分。

---

## 四模式與歷史評估

`rendering.profiles` 分別設定 standard、natural、cinematic、neutral。預設採用使用者指定的結構／自動控制／強度組合；neutral 明確指定 style 2，其餘模式跟隨本身 style。`tone: null` 跟隨模式預設。`dataset.render_pass` 固定 clean，motion 固定 Sintel，不提供 control_mask。`temporal: false` 使用單張，`true` 支援最多 10 幀；資料載入器只打亂片段，不打亂片段內幀序，不跨場景或斷幀。

增強保留「噪聲＋隨機一種亮度／對比／銳利度」。每個有效 batch 同時限制片段與幀的增強比例，不超過排程比例；其餘保持原圖。強度從 10% 升至 25%，50% epoch 達最高，75% 起關閉。片段內維持相同模式與強度；Teacher 始終使用未增強影像。

四模式在相同幾何下約產生四倍目標；`releas: true` 仍先建立／續建快取並完整釋放 Teacher，`false` 維持凍結 Teacher 的線上蒸餾。舊平面快取保留但不重用。

`temporal_latest.json/.csv/.png` 列出各 profile 第 1～10 幀的 loss、PSNR、SSIM、LPIPS，以及同張圖重置歷史的分數與差值。兩者對照相同的 temporal Teacher RGB 目標，另記錄實際歷史深度、使用率、有效像素與光流對齊 MAE；這些是 Teacher 一致性與時間穩定性指標，不是獨立清晰 GT 品質。獨立驗證輸出 `temporal_val`／`temporal_test` 報表。

圖像輸出保留 preview／pred，預測路徑加入 profile：`pred/<student|teacher>/<profile>/<mode>/...png`。每種增強模式抽選兩個片段的末幀輸出，先依序推論歷史上下文，再輸出該末幀比較圖。total／avg infer 包含實際推論的上下文幀，單位秒。只有 RGB 的部署影片需要另外由 RGB 估計光流，不能使用 Sintel GT。

---

## License

專案程式採用 [MIT](LICENSE)。外部相依套件、資料集及 Teacher 權重各依其原有條款。
