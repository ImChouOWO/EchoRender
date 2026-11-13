# EchoRender

[English](README.md)

![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange) ![Task](https://img.shields.io/badge/Task-RGB_SR-blue) ![Precision](https://img.shields.io/badge/Precision-BF16-green)

以 REDS LR／HR 配對訓練 RGB 超解析度，並依序監督提升解析度、去模糊、去噪。新版已移除 Teacher、蒸餾與特徵快取，不支援舊版蒸餾權重。

```text
當前 LR RGB＋光流對齊的前幀 LR RGB
→ 6 通道瓶頸／Swin adapter → Encoder／SSM／ViT／Decoder
→ 遮罩殘差交叉注意力超解析度 → NAFNet → Restormer → 最終 HR RGB
```

<details><summary><em>當前限制 / Current Limitations</em></summary>

新的超解析度頭需要重新訓練，舊 PixelShuffle SR 與蒸餾權重均不相容。合成 Gaussian 模糊與噪聲不能代表真實 CCTV 的所有退化。原生尺寸推論使用重疊 tiles，比較績效時必須維持相同協定。模組規模可設定，不宣稱重現上游論文績效。

---

New training is required. Synthetic Gaussian degradations do not establish real CCTV performance. Compare scores using identical tile/context settings.

</details>

---

## 簡介

保留原有 Encoder／Decoder 跳躍連接與 InputAdapter 到輸出端的特徵旁路。影像不再全圖壓成固定方形；訓練改用全圖範圍內隨機取樣、LR／HR 對齊的 patches。

| 模組 | 功能 |
|---|---|
| 輸入 | 當前 RGB 3 通道＋對齊的歷史 RGB 3 通道，無 profile／tone |
| 光流 | 保留 `raft_small` 為預設並凍結作 RGB 對齊；可選 `raw` 外部 Farneback 或 `feat` RGB 配對 CNN |
| SSM | 官方 Mamba CUDA selective scan，遞迴使用 FP32 |
| 排序 | 重疊視窗內依 router 分組，保留方向輪轉，scan 後還原排列並融合重疊區域 |
| 參數 | 共用投影後依 `[dt_rank, state_dim, state_dim]` 切成 delta、B、C |
| Prompt | 可學習字典＋排序特徵與三個參數 chunk 的 1×1 投影 |
| 先驗 | 零初始化的可學習 feature map，不是外部提供的語意標籤 |
| 重建 | 兩級 2× 遮罩殘差交叉注意力組成 4× SR，接續 NAFNet 去模糊與 Restormer 去噪 |

每級 2× SR 保留原有特徵，只查詢三個新增位置。注意力採獨立 Q／K／V、四個 heads、8×8 視窗與 0／4 位移。共享 MLP 只在 bicubic 基底的新增位置加入 RGB 殘差。4× 監督仍接在第二級輸出。

SSM 基本設定參考 MambaIRv2-Small：state=16、expand=2、routing groups=128、prompt rank=64；視窗重疊與參數條件 prompt 為本專案設計。

---

## 專案結構

```text
config/model.yaml       模型 DAG、SSM、motion、三階段輸出頭
config/train.yaml       訓練、REDS 配對、合成、驗證
model/card/             模型元件與 loss
train.py                訓練與續訓
val.py                  驗證與範例匯出
infer.py                單張／最多 10 幀推論
preview.py              比較頁與訓練績效頁
output/                 所有產生的結果
```

---

## 安裝

### 1. 啟用環境

```bash
cd /home/soic/Desktop/EchoRender
uv sync --locked
source .venv/bin/activate
```

### 2. 驗證

```bash
python -m unittest tool.test_sr -v
```

GPU 使用已匹配 PyTorch／CUDA 的官方 Mamba extension；卷積、線性層與注意力可用 BF16，scan 與 RGB loss 使用 FP32。

---

## 資料集格式

```text
REDS/train/train_sharp/000/00000000.png
REDS/train/train_sharp_bicubic/X4/000/00000000.png
REDS/val/val_sharp/000/00000000.png
REDS/val/val_sharp_bicubic/X4/000/00000000.png
```

同場景、同檔名配對；支援官方解包後省略 `X4` 的目錄結構。範例設定先選 10 個訓練場景、2 個驗證場景，`scenes: null` 表示全部。每個 clip 最多 10 幀，光流只從 RGB 計算，不使用不存在的 REDS 光流 GT。

---

## 合成與監督

乾淨 HR 為 `G`，模糊為 `B(G)`；LR 噪聲為 `N`，以 bilinear 對應到 HR 的同一份噪聲為 `U(N)`：

| 階段 | 答案 |
|---|---|
| SR | `B(G) + U(N)` |
| 去模糊 | `G + U(N)` |
| 去噪／最終輸出 | `G` |

退化 LR 為 `downsample(B(G)) + N`。噪聲先在 LR 產生，再映射到 HR 中間答案，避免要求 SR 重建已被降採樣消除的獨立隨機噪聲。模糊 sigma 以 LR 像素為單位，合成 HR 時按放大倍數換算。中間答案不裁切數值，保留相同噪聲的代數關係；實際 LR 輸入限制於 [0,1]。未增強資料使用 REDS 原有 LR，三階段答案皆為乾淨 HR。

排程沿用 10% 起始、總 epoch 50% 時達 25%、75% 時關閉。比例同時控制強度與 batch 內增強 clip 上限，數量採 `floor(batch × ratio)`；小 batch 可能無增強，始終保留未增強資料。增強僅剩模糊與噪聲。

驗證範例輸出固定最大強度的 25%／50%／75%／100%。Preview 另提供直接對 LR 加入干擾，以便檢查部署輸入；介面會標註此差異。

---

## 設定

| 參數 | 意義 |
|---|---|
| `runtime.output_dir` | 僅能位於專案 output 下，重名自動加 `_2` 起的流水號 |
| `dataset.lr_patch_size` | 預設 80，×4 對應 HR 320；隨機覆蓋全圖 |
| `loader.batch_size` | 每次 optimizer 更新的有效 clip 數 |
| `training.micro_batch_size` | 同時運算 clip 數，其餘透過梯度累積 |
| `model.yaml: motion` | `raft_small`（預設）／`raw`／`feat`；REDS 無光流 GT |
| `model.yaml: ssm` | 方向、視窗、stride、router 與 prompt 規模 |
| `model.yaml: reconstruction` | 放大倍數及 NAFNet／Restormer 規模 |
| `validation.tile_size/tile_overlap` | LR 推論 tile 與重疊，輸出仍為原生 HR 尺寸 |
| `dataset.splits.test` | 預設 null，官方 REDS test 未公開 HR GT |

---

## Loss 與排程

```text
L = w_sr × L1(Y1, B(G)+U(N))
  + w_deblur × [Charbonnier(Y2, G+U(N)) + 0.05 × Edge(Y2, G+U(N))]
  + w_denoise × L1(Y3, G)
```

三階段權重暫設 1，皆可調整。Edge 採 MPRNet 的 Laplacian residual 定義，不作為自創驗證指標。每個 clip 按幀數正規化，避免長 clip 主導更新。歷史資訊使用前一幀退化 LR，跨幀不保留反向傳播圖。

維持 SGD：LR=0.001、momentum=0、weight_decay=0。每次 optimizer 更新前調整 cosine，首個更新使用初始 LR、最後使用 eta_min=0.000001；只有一次更新時使用初始 LR。無 warmup／restarts。

---

## 操作

```bash
python train.py --config config/train.yaml
python train.py --config config/train.yaml --resume output/sr/last.pt
python val.py --weights output/sr/best.pt --project sr_validation
python infer.py --weights output/sr/best.pt --input image.png --device cuda:1
python preview.py --ip 127.0.0.1 --port 8092
```

僅保留 `best.pt`、`last.pt` 兩個固定權重名稱。續訓從最近完成的 epoch 還原 optimizer、RNG、loader RNG、排程、history、早停；未完成的 epoch 會重跑。需要相同模型／訓練設定與原有 best.pt。終端機保留固定欄位、GPU 使用率與最佳權重 highlight。

---

## 驗證與輸出

使用公認 PSNR↑、SSIM↑、LPIPS↓，答案為乾淨 HR。預設 REDS RGB、crop=0、逐圖平均、重疊 tile 推論；可明確設定 Y-channel 與 border。LPIPS 始終使用 RGB。RGB／梯度誤差熱力圖僅供診斷，不列為新增指標。

所有結果位於 `output/`：訓練權重與 TensorBoard 位於各 run；summary、CSV、曲線及最佳權重範例位於 `output/val/<run>/`。預覽、預測與熱力圖分開存放，各模式與影像名稱獨立分組，每個強度一張完整比較圖。

Preview 頁支援路徑／上傳、權重、GPU、幀數、tile、干擾控制、三階段結果、原圖／GT 拉桿比較與原尺寸查看。Training 頁支援 loss／指標走勢與標準 TensorBoard。

訓練完成列印 loss、PSNR、SSIM、LPIPS 的 min／max／mean／best；summary 另記錄選定最佳 checkpoint 時的各指標。

---

## 參考資料

- [REDS](https://seungjunnah.github.io/Datasets/reds.html)
- [MambaIR](https://github.com/csguoh/MambaIR)、[Mamba](https://github.com/state-spaces/mamba)
- [NAFNet](https://github.com/megvii-research/NAFNet)、[Restormer](https://github.com/swz30/Restormer)
- [MPRNet](https://github.com/swz30/MPRNet)

## License

見 [LICENSE](LICENSE) 與 [THIRD_PARTY.md](THIRD_PARTY.md)。
