# 🛰️ Full Master Plan: GISTDA PM2.5 Nationwide Benchmarking Suite (618 Stations × 24 Hours)

> **เอกสารแผนงานทางเทคนิค**: แผนการพัฒนาสคริปต์ Benchmark โมเดลทำนายฝุ่นดาวเทียมเทียบกับ GISTDA PM2.5 API ทั่วประเทศไทย  
> **ที่ตั้งไฟล์**: `ai-docs/gistda_benchmark_plan.md`  
> **เป้าหมาย**: ประเมินความแม่นยำของโมเดล Machine Learning / Stacking Ensemble จากโฟลเดอร์โมเดล (`outputs/pm25/models/`) เปรียบเทียบกับข้อมูลสังเกตการณ์จริงรายชั่วโมงย้อนหลัง 24 ชั่วโมงจาก **GISTDA PM2.5 REST API** ครอบคลุม 618 สถานีตรวจวัดทั่วประเทศ (~14,800 แถว) โดยทำงานแบบ Standalone ไม่พึ่งพา `run_pipeline.py`

---

## 1. บทนำและบริบทของระบบ (Executive Context & Objectives)

### 1.1 วัตถุประสงค์
1. **ล็อกข้อมูลภาคพื้นดิน GISTDA ทันที**: เนื่องจาก GISTDA REST API เก็บข้อมูลแบบ Rolling Window เพียง **24 ชั่วโมงย้อนหลัง** เท่านั้น หากไม่ดึงมาบันทึกเก็บไว้ ข้อมูลจะถูกเลื่อนทับชั่วโมงต่อชั่วโมง
2. **ขอบเขตการทดสอบระดับประเทศ**: ดึงข้อมูลจาก 618 พิกัดสถานีสังเกตการณ์ทั่วประเทศไทย ($618 \times 24 \approx 14,832$ จุดข้อมูล) เพื่อให้ได้กลุ่มตัวอย่างขนาดใหญ่ระดับสถิติ (~12k - 15k rows)
3. **การทดสอบแบบ Standalone และโมดูลาร์**: ไม่รันผ่าน `run_pipeline.py` เพื่อลด Overhead และหลีกเลี่ยงการเทรนโมเดลซ้ำ โดยโหลดโมเดลที่เทรนเสร็จแล้วจากไดเรกทอรีโมเดล (`outputs/pm25/models/<experiment>/` เช่น `exp_2km_pm25` หรือ `exp_1km_pm25`) มาทำ Inference โดยตรง
4. **การจัดการ Timezone UTC+7 ที่แม่นยำ**: ปรับเวลาท้องถิ่นประเทศไทย (UTC+7) ให้เป็น UTC สากล เพื่อดึงภาพถ่ายดาวเทียม Himawari-9 จาก AWS S3 ให้ตรงชั่วโมงสังเกตการณ์จริงอย่างสมบูรณ์

---

## 2. แผนภาพสถาปัตยกรรมระบบ (End-to-End System Architecture)

```mermaid
flowchart TD
    subgraph Step1 ["Phase 1: Ingestion ข้อมูล GISTDA 24h Ground Truth"]
        A["data/metadata/openaq_thailand_stations.csv<br/>(618 สถานีพิกัด Lat/Lng)"] -->|Concurrent ThreadPool (8-10 workers)| B("GISTDA REST API<br/>/rest/getPm25byLocation?lat={lat}&lng={lng}")
        B --> C["Parse 24h Graph History<br/>Timezone Normalization (UTC+7 -> UTC)"]
        C --> D[("data/benchmark/gistda_ground_truth_24h.parquet<br/>(~14,800 rows)")]
    end

    subgraph Step2 ["Phase 2: Ingestion ภาพถ่ายดาวเทียม Himawari-9"]
        D -->|สกัด 24 Unique UTC Timestamps| E["NOAA AWS S3 Open Data<br/>(s3://noaa-himawari9/)"]
        E -->|Batch Download 24 Scenes Only| F["HimawariIRLoader<br/>(Segments 3, 4, 5 / 10 IR Bands)"]
        F --> G["SatelliteTabularizer<br/>(Centroid BTs, BTDs, 3x3 Texture)"]
    end

    subgraph Step3 ["Phase 3: Spatial Feature Fusion & Alignment"]
        H[("thailand_2km_master_grid.parquet<br/>(DEM, Slope, Aspect, Coast, Coriolis)")] -->|KDTree Positional Lookup| I["Unified Feature Matrix<br/>(52+ Multimodal Columns)"]
        G --> I
        J["Solar Angles & Cyclical Time Engines<br/>(SZA, is_daytime, hour_sin/cos, month_sin/cos)"] --> I
    end

    subgraph Step4 ["Phase 4: Direct Model Inference (Dir-Out)"]
        K[("outputs/pm25/models/exp_2km_pm25/<br/>Ensemble & Base Checkpoints")] -->|DustStackingEnsemble.load| L["Direct Inference Engine<br/>(No run_pipeline dependency)"]
        I --> L
        L --> M["Prediction Generation<br/>(pred_ensemble, pred_catboost, pred_xgboost, etc.)"]
    end

    subgraph Step5 ["Phase 5: Benchmark Evaluation & Deliverables"]
        D --> N["Evaluation Engine"]
        M --> N
        N --> O["Continuous Metrics<br/>(MAE, RMSE, MAPE, Pearson r, R², MBE)"]
        N --> P["PCD Thailand AQI 5-Tier Classification<br/>(Accuracy, Macro-F1, Cohen's Kappa, Confusion Matrix)"]
        N --> Q[("Deliverables<br/>• benchmark_predictions.parquet<br/>• gistda_benchmark_report.json / .md<br/>• Parity Scatter & Residual Plots")]
    end
```

---

## 3. ประเด็นวิกฤตทางเทคนิค (Critical Engineering Considerations)

### 3.1 การจัดการ Timezone (+07:00 vs UTC)
จากการตรวจสอบ Response ของ GISTDA API:
```json
{
  "status": 200,
  "data": {
    "datetimeEng": {"dateEng": "Thursday 15 October 2026", "timeEng": "3:00"},
    "graphHistory24hrs": [
      [21.992401, "2026-10-14T01:00:00.000Z"],
      ...,
      [7.7856236, "2026-10-15T03:00:00.000Z"]
    ]
  }
}
```
* **ข้อเท็จจริง**: ข้อมูลชุดล่าสุดระบุเวลา `datetimeEng: 3:00` ซึ่งตรงกับเวลาไทย (03:00 น. วันที่ 15 ต.ค.) ขณะที่ในอาเรย์ `graphHistory24hrs` ส่งสตริง `"2026-10-15T03:00:00.000Z"` มาพร้อมตัวอักษร `Z` (Zulu/UTC)
* **ข้อควรระวัง**: หากใช้ `pd.to_datetime(ts, utc=True)` ระบบจะตีความว่าเป็น 03:00 UTC ซึ่งจะเร็วกว่าเวลาจริง 7 ชั่วโมง (กลายเป็น 10:00 น. เวลาไทยในอนาคตที่ยังไม่มีไฟล์ดาวเทียมจริงบน S3)
* **อัลกอริทึมการแปลงที่ถูกต้อง**:
  ```python
  # 1. ตีความตัวเลขเวลาเป็นเวลาประเทศไทย (Asia/Bangkok)
  dt_bkk = pd.to_datetime(ts_str[:19]).tz_localize("Asia/Bangkok")
  # 2. แปลงเป็น UTC สำหรับจับคู่ไฟล์ดาวเทียม
  dt_utc = dt_bkk.tz_convert("UTC")
  ```
  *ผลลัพธ์*: `2026-10-15 03:00 BKK` $\rightarrow$ `2026-10-14 20:00 UTC` ซึ่งตรงกับโฟลเดอร์ภาพดาวเทียมบน S3 `AHI-L1b-FLDK/2026/10/14/2000/` อย่างถูกต้อง 100%

### 3.2 การเพิ่มประสิทธิภาพการดึงภาพดาวเทียม (Batch 24 Scenes Optimization)
- ทั้ง 618 สถานีตรวจวัดใช้ช่วงเวลาสังเกตการณ์ในรอบ **24 ชั่วโมงชุดเดียวกัน**
- แทนที่จะดึงภาพแยกตามสถานี ($618 \times 24 = 14,832$ ครั้ง) สคริปต์จะจัดกลุ่มดึงภาพ Himawari-9 **เพียง 24 Scenes (24 UTC Timestamps)**
- จากแต่ละ Scene จะทำการ Tabularize พิกัดทั้ง 618 สถานีพร้อมกันด้วยการคำนวณแบบ Vectorized ผ่าน KDTree ทำให้ลดแบนด์วิดท์และเวลาดาวน์โหลดลงกว่า **99.8%**

---

## 4. แผนการดำเนินงานรายเฟสอย่างละเอียด (Phased Implementation Plan)

### Phase 1: Ingestion & Caching ข้อมูล GISTDA 24h Ground Truth
1. **ไฟล์ต้นทางพิกัดสถานี**:
   - อ่านจาก `data/metadata/openaq_thailand_stations.csv` (มี 618 สถานีทั่วประเทศ ครอบคลุมพิกัด `latitude`, `longitude`, `station_code`, `station_name`)
2. **การยิง API แบบ Concurrent**:
   - ใช้ `ThreadPoolExecutor(max_workers=8..10)` พร้อม Rate Limiter หน่วงเวลาเล็กน้อยเพื่อป้องกันการถูกตัดการเชื่อมต่อ
   - Endpoint: `https://pm25.gistda.or.th/rest/getPm25byLocation?lat={lat}&lng={lon}`
   - ติดตั้ง Exponential Backoff Retry (3 ครั้ง) กรณีเกิด Network Timeout
3. **การแตกข้อมูลและจัดเก็บ**:
   - แตกอาเรย์ `graphHistory24hrs` (24 ค่าต่อสถานี)
   - ฟิลด์ที่บันทึก:
     - `station_code`, `location_id`, `station_name`, `latitude`, `longitude`
     - `timestamp_bkk` (Asia/Bangkok)
     - `timestamp_utc` (UTC สากล)
     - `gistda_pm25` (ความเข้มข้น PM2.5 $\mu\text{g/m}^3$)
4. **Data Validation & Persistence**:
   - กรองค่าติดลบหรือค่าผิดปกติทางสรีรวิทยา ($0 \le \text{PM}_{2.5} \le 500\,\mu\text{g/m}^3$)
   - บันทึกตารางลง `data/benchmark/gistda_ground_truth_24h.parquet` และสำเนา `.csv`

---

### Phase 2: Ingestion & Tabularization ภาพถ่ายดาวเทียม Himawari-9
1. **ระบุช่วงเวลาดาวเทียม**:
   - คัดเลือกรายชื่อ `timestamp_utc` ที่ไม่ซ้ำกันจาก Phase 1 (24 ช่วงเวลา)
2. **ดาวน์โหลดจาก NOAA AWS S3 (`noaa-himawari9`)**:
   - ใช้โมดูล [HimawariIRLoader](file:///e:/water-analysis-project/dust-prediction-satelite/src/data/himawari_ir_loader.py)
   - แถบคลื่น: 5 Core IR Bands (`B07`, `B10`, `B11`, `B13`, `B15`) และ Infrared เสริม (`B08`, `B09`, `B12`, `B14`, `B16`)
   - เซกเมนต์: Segment 4 (ครอบคลุมพื้นที่ 92.9% ของประเทศ) ควบคู่กับ Segments 3 และ 5 สำหรับภาคใต้
3. **สกัดฟีเจอร์รังสีเชิงกายภาพ**:
   - ใช้โมดูล [SatelliteTabularizer](file:///e:/water-analysis-project/dust-prediction-satelite/src/features/satellite_tabularizer.py) สกัดฟีเจอร์ที่พิกัด 618 สถานี:
     - Centroid BTs: `bt_b07` ถึง `bt_b16` (Kelvin)
     - Physics BTDs: `btd_15_13` (ฝุ่นซิลิเกต), `btd_13_07` (จุดความร้อน/ควันชีวมวล), `btd_14_11`, `btd_10_08`, `dust_rgb_red`
     - Spatial Texture $3\times3$: `b13_mean_3x3`, `b13_std_3x3`, `b15_mean_3x3`, `btd_15_13_mean_3x3`
     - Temporal Dynamics: `delta_bt13_cadence`, `delta_btd15_13_cadence`

---

### Phase 3: Spatial Feature Fusion & Topographic/Temporal Alignment
เพื่อให้ตารางฟีเจอร์สอดคล้องกับ 52+ ฟีเจอร์ที่โมเดลเทรนไว้:
1. **ฟีเจอร์ภูมิประเทศ (Copernicus DEM 30m)**:
   - จับคู่พิกัด 618 สถานีเข้ากับ [thailand_2km_master_grid.parquet](file:///e:/water-analysis-project/dust-prediction-satelite/data/geo/thailand_2km_master_grid.parquet) ด้วย Scipy KDTree
   - เติมค่า: `elevation_m`, `elevation_std`, `slope_deg`, `aspect_sin`, `aspect_cos`, `dist_coast_km`, `coriolis_param`, `tpi_index`, `relief_m`
2. **ฟีเจอร์แสงอาทิตย์และวงรอบเวลา (Solar & Temporal Cyclical)**:
   - คำนวณผ่าน [grid_engine.py](file:///e:/water-analysis-project/dust-prediction-satelite/src/features/grid_engine.py):
     - `solar_zenith_angle` (มุมองศาดวงอาทิตย์)
     - `is_daytime` (1 เมื่อมีแสงอาทิตย์, 0 เมื่อเป็นกลางคืน)
     - `hour_sin`, `hour_cos` (วงรอบ 24 ชั่วโมง)
     - `month_sin`, `month_cos` (ฤดูกาล)
3. **ฟีเจอร์สภาพอากาศ (Weather Proxies / Climatology)**:
   - เติมค่าความกดอากาศและความชื้นสัมพัทธ์ตามโครงสร้าง Climatological Fallback เพื่อให้ชุดข้อมูลสมบูรณ์ปราศจาก Missing Values

---

### Phase 4: Direct Model Inference (Dir-Out Standalone Engine)
1. **การโหลดโมเดล**:
   - ใช้โมดูล [DustStackingEnsemble](file:///e:/water-analysis-project/dust-prediction-satelite/src/models/ensemble_model.py) โหลดตรงจากไดเรกทอรีที่กำหนด:
     ```python
     ensemble = DustStackingEnsemble.load(model_dir)
     ```
   - สคริปต์จะตรวจสอบความเข้ากันได้ของคอลัมน์ฟีเจอร์ (`ensemble.feature_names_`)
2. **การทำนายผล**:
   - ทำนายค่าฝุ่นระดับชั่วโมง:
     - `pred_ensemble`: ผลลัพธ์ Stacking Meta-Learner ถ่วงน้ำหนัก
     - `pred_catboost`, `pred_xgboost`, `pred_lightgbm`, `pred_random_forest`: ผลลัพธ์จากแต่ละโมเดลย่อยเพื่อเปรียบเทียบความแม่นยำรายอัลกอริทึม
3. **รวมผลเข้าสู่ตาราง Benchmark**:
   - แนบคอลัมน์ทำนายคู่กับ `gistda_pm25` พร้อมคำนวณ `residual = pred_ensemble - gistda_pm25` และ `error_abs = |residual|`

---

### Phase 5: Comprehensive Statistical Benchmark & Evaluation Metrics
1. **ตัวชี้วัดความแม่นยำทางสถิติแบบต่อเนื่อง (Continuous Metrics)**:
   - **MAE (Mean Absolute Error)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$ ($\mu\text{g/m}^3$)
   - **RMSE (Root Mean Square Error)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$ ($\mu\text{g/m}^3$)
   - **MAPE (Mean Absolute Percentage Error)**: $\frac{100\%}{N} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|$
   - **Pearson Correlation ($r$)**: วัดทิศทางความสอดคล้องระหว่างดาวเทียมกับ GISTDA
   - **Coefficient of Determination ($R^2$)**: ความสามารถในการอธิบายความแปรปรวน
   - **MBE (Mean Bias Error)**: $\frac{1}{N} \sum (\hat{y}_i - y_i)$ บ่งชี้ว่าโมเดลทำนายสูงเกิน (Overprediction) หรือต่ำเกิน (Underprediction)
2. **ตัวชี้วัดการจัดเกรดคุณภาพอากาศ (Thailand PCD AQI 5-Tier Classification)**:
   - จำแนกฝุ่นตามมาตรฐานกรมควบคุมมลพิษ:
     * ฟ้า (0.0 – 15.0): คุณภาพอากาศดีมาก
     * เขียว (15.1 – 25.0): คุณภาพอากาศดี
     * เหลือง (25.1 – 37.5): คุณภาพอากาศปานกลาง
     * ส้ม (37.6 – 75.0): เริ่มมีผลกระทบต่อสุขภาพ
     * แดง (> 75.0): มีผลกระทบต่อสุขภาพ
   - คำนวณ: **Overall Accuracy**, **Macro F1-Score**, **Cohen's Kappa**, และ **Confusion Matrix**
3. **การวิเคราะห์มิติเชิงลึก (In-Depth Diagnostics)**:
   - **Temporal Breakdown**: กราฟ MAE รายชั่วโมง เทียบประสิทธิภาพช่วงกลางวัน vs กลางคืน
   - **Regional Breakdown**: แยกวิเคราะห์ความแม่นยำ 6 ภูมิภาค (เหนือ, ตะวันออกเฉียงเหนือ, กลาง, ตะวันออก, ตะวันตก, ใต้)
4. **บันทึกผลลัพธ์และรายงาน**:
   - `outputs/pm25/evaluation/gistda/gistda_benchmark_report.json`
   - `outputs/pm25/evaluation/gistda/gistda_benchmark_report.md`
   - `outputs/pm25/evaluation/gistda/benchmark_predictions.parquet`
   - แผนภาพการกระจายตัว Parity Scatter Plot (`parity_gistda_vs_satellite.png`) และกราฟ Residual Time-series

---

## 5. อินเตอร์เฟซคำสั่ง CLI (`benchmark_gistda.py`)

```bash
# 1. แนะนำ: ดึงข้อมูล GISTDA เก็บไว้ก่อนทันทีเพื่อป้องกันข้อมูลหลุดกรอบ 24h
python benchmark_gistda.py --step download-gistda

# 2. ทำการดึงภาพดาวเทียมและทดสอบเทียบกับโมเดลโดยใช้ข้อมูล GISTDA ที่แคชไว้แล้ว
python benchmark_gistda.py --step run-benchmark --use-cached-gistda --model-dir outputs/pm25/models/exp_2km_pm25

# 3. รันครบทุกขั้นตอนแบบ All-in-one ตั้งแต่ดาวน์โหลดจนถึงออกรายงานสรุป
python benchmark_gistda.py --model-dir outputs/pm25/models/exp_2km_pm25

# 4. ทดสอบเปรียบเทียบกับโมเดลความละเอียดอื่น (เช่น 1km หรือ 500m)
python benchmark_gistda.py --model-dir outputs/pm25/models/exp_1km_pm25 --use-cached-gistda
```

### รายละเอียด Argument Flags:
| Argument | Type | Default | คำอธิบาย |
| :--- | :---: | :---: | :--- |
| `--step` | `str` | `all` | เลือกขั้นตอน: `all`, `download-gistda`, `run-benchmark` |
| `--model-dir` | `Path` | `outputs/pm25/models/exp_2km_pm25` | โฟลเดอร์โมเดลที่ต้องการนำมาประเมินผล |
| `--stations-csv` | `Path` | `data/metadata/openaq_thailand_stations.csv` | ไฟล์พิกัด 618 สถานี |
| `--output-dir` | `Path` | `outputs/pm25/evaluation/gistda` | โฟลเดอร์จัดเก็บบันทึกผลลัพธ์และรายงาน |
| `--use-cached-gistda` | `flag` | `False` | ใช้งานข้อมูล GISTDA ที่เคยโหลดไว้แล้ว ไม่ต้องยิง API ซ้ำ |
| `--use-cached-sat` | `flag` | `False` | ใช้งานไฟล์ภาพดาวเทียมในแคชเดิม |
| `--max-workers` | `int` | `8` | จำนวน Thread ยิง API และดาวน์โหลด S3 พร้อมกัน |

---

## 6. สรุปความพร้อมทรัพยากรในระบบ (System Readiness Checklist)

- [x] **พิกัดสถานี 618 แห่ง**: มีพร้อมใน `data/metadata/openaq_thailand_stations.csv`
- [x] **GISTDA REST API Endpoint**: ทดสอบการยิงแล้ว สถานะ HTTP 200 และดึง `graphHistory24hrs` ได้ถูกต้อง
- [x] **NOAA Himawari-9 S3 Bucket**: ทดสอบ Bucket `noaa-himawari9` พบว่ามีไฟล์สังเกตการณ์พร้อมในรอบเวลาจริง
- [x] **Master Grid & DEM**: `data/geo/thailand_2km_master_grid.parquet` สมบูรณ์พร้อมเชื่อมโยง KDTree
- [x] **Ensemble Checkpoints**: ทดสอบ `DustStackingEnsemble.load()` จากโฟลเดอร์โมเดลผ่านฉลุย
- [ ] **การสร้างสคริปต์ `benchmark_gistda.py`**: พร้อมเข้าสู่ขั้นตอนเขียนโค้ดและรัน Ingestion ทันที
