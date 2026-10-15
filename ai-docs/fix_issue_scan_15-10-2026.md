# Fix Issue Scan — 15-10-2026 (ยึด `src/` เป็นหลัก ไม่สน md)

 scope: `src/config`, `src/data`, `src/features`, `src/models`, `src/evaluation` + `configs/*.yaml` + `run_pipeline.py`
 โจทย์: dust nowcast จาก satellite weather image, PM2.5/PM10 ไทย, ใช้ได้กลางวัน+กลางคืน, พึ่งภาคพื้นน้อยที่สุด (pure-sat vs non-pure)
 train ใช้เฉพาะ cell ที่มีสถานีฝุ่น (match KDTree แล้ว `filter_training_cells` ตัด NaN ทิ้ง) — ยืนยันจาก code แล้ว

---

## A. คำสั่งแก้ time/location leak (ทำก่อน)

### A1. ตัด `month/hour/sza` ออกจาก training feature
- ปัจจุบันสอนตรง: `feature_matrix_builder.py:137-153` `TOPO_SOLAR_COLUMNS` มี `solar_zenith_angle, is_daytime, hour_sin, hour_cos, month_sin, month_cos`
- เติมโดย `grid_engine.py:517-523` `attach_solar_and_time_features()`
- เข้าโมเดลทั้ง pure/full ผ่าน `ALL_FEATURE_COLUMNS:179-187` → `FULL_MULTIMODAL:190`, `NO_WEATHER:169-176` → `trainer.py:120,464-465`, `hyperparameter_tuner.py:47-48`
- **แก้:** เอา 6 ตัวนี้ออกจาก `TOPO_SOLAR_COLUMNS` คงไว้แค่ `elevation_m, tpi_index, slope_deg, aspect_sin/cos, elevation_std, relief_m, dist_coast_km`
- `sza` เก็บไว้แค่ (1) stratify/eval day-night (2) physics-correct `btd_13_07` ใน `satellite_tabularizer.py:79` ด้วย `cos(SZA)` แล้วทิ้ง ไม่ให้เข้าโมเดล หรือแยกโมเดล day/night แต่ข้างในไม่มี SZA

### A2. แก้ lat รั่วทางอ้อม
- `lat/lon/utm` ดิบไม่เข้าโมเดล (อยู่แค่ `KEY_COLUMNS:76`, exclude ใน `trainer.py:123`, แยก keys/features ใน `build_snapshot_matrix:886-887`) — ดีแล้ว ห้ามแก้ให้รั่ว
- แต่รั่ว 3 ทาง:
  1. `coriolis_param = 2Ω·sin(lat)` (`grid_engine.py:149,452`) — **ตัดทิ้ง** ไม่มีผลฟิสิกส์ในสเกล 2km
  2. `dist_coast_km/elevation/tpi` เป็น location fingerprint — เก็บได้แต่ต้องกันจำสถานีด้วย target-anomaly (`y - monthly_station_climatology`) + `GroupKFold by station/province` เป็น metric หลักแทน `cv=1`
  3. `hour/month` คือ time-fingerprint ของ lat — โดนตัดไปพร้อม A1 แล้ว
- offshore (`dist_coast` ใหญ่ / elevation 0) คือ out-of-domain → บังคับ pure-sat ห้ามใช้ full-mode ทะเล

---

## P0 — ผิดแล้วเลขหลอก (ทำก่อนอ้างผล)

- **P0-1 split รั่วเวลา+สถานที่:** `model_config.yaml:10` `cv_splits:1` → `ensemble_model.py:134-140` shuffle row-level 85/15, `hyperparameter_tuner.py:129-136` `split_holdout` แบบเดียวกัน, `ensemble_model.py:229` `KFold(shuffle=True)` ไม่มี grouping → ชั่วโมงติดกัน+สถานีเดียวกันอยู่ทั้ง train/val
  - แก้: `GroupKFold by station_id + Out-of-Time 2024-25/2026` เป็นเลขหลัก
- **P0-2 imputer ไม่เคย fit:** `feature_matrix_builder.py:883` เรียก `impute_features()` ตรงๆ `fitted_stats_={}` → fallback median ของ batch นั้น (`:388-392,:494-497`) = เอา test มา impute test + train-serve skew
  - แก้: `fit(train)` ครั้งเดียวแล้ว `transform()` ทุกชุด
- **P0-3 flag หาย:** aux นิยาม 3 ตัว (`:162-166`) แต่ `build_snapshot_matrix:886-888` เก็บแค่ `cloud_occluded_flag, imputed_flag` ทิ้ง `temporal_delta_valid` (`:488`)
  - แก้: ใส่กลับเข้า `ordered_cols`
- **P0-4 cloud fill ฆ่าสัญญาณ:** `:372-393` `bt_b13<220K|NaN → fill median ~290K` เมฆหนากลายเป็นฟ้าใสปลอม
  - แก้: ห้าม fill BT เมฆ ให้เรียนจาก flag+BTD ตรงๆ
- **P0-5 B07 day/night คนละ physics:** `satellite_tabularizer.py:79` `btd_13_07` ลบดิบ กลางวันปน solar กลางคืน thermal ล้วน ไม่มี SZA interaction ไม่มี eval แยก day/night ใน `metrics/spatial_cv/benchmark_suite`
  - แก้: ดู A1 (correct-then-drop + รายงาน day vs night ทุก run + ablation ตัด B07)
- **P0-6 metric ปั้นเลข:** `metrics.py:239` ไม่มี high-risk sample → return `100.0%`
  - แก้: return `NaN`

---

## P1 — ขวางโจทย์ pure-sat / พึ่งพื้นน้อยที่สุด

- **P1-1 full-mode ทะเลคือ climatology:** `spatial_directional.py:39,113-123` เกิน 50km ได้ `{}` → NaN → impute 1013.25/70% (`feature_matrix_builder.py:490-538`) ไม่มี router เลือก pure-sat อัตโนมัติ; 1-quadrant pressure clip `[0.2,5.0]` (`:240`) แต่ ≥2 quadrants ไม่ clip; humidity 1-quadrant ตกเป็น NaN (`:279-284`)
- **P1-2 BTD alias ซ้ำซ้อน + schema drift:** `satellite_tabularizer.py:93-100` alias `btd_14_11=btd_13_11`, `btd_10_08=btd_13_10` เงียบๆ แล้ว `trainer.py:132-137` prune ด้วย `allclose` แค่ 100 แถว; `feature_matrix_builder.py:402-404` สร้าง `dust_rgb_red` ใหม่ทั้งที่ `SAT_BTD_COLUMNS:84-89` ตัดออกแล้ว
- **P1-3 texture/past-frame:** `satellite_tabularizer.py:368-386` reuse `target_rows/cols` ข้าม timestamp; `extract_3x3_texture:155-193` ขอบภาพ window แหว่ง variance เพี้ยน
- **P1-4 objective ไม่ตรง + physics ปิด:** code default `regression_l1/absoluteerror/MAE` แต่ yaml override Tweedie 1.5 ทั้งแผง, `monotonic_constraints:{}` ว่าง → constraint ใน `lgbm:74-80/xgb:70-73/catboost:66-69` ไม่ทำงาน
- **P1-5 HII fallback พังเงียบ:** `hii_weather_loader.py:85-91` metadata หาย → ยัด 20 stations พิกัดเดียว `(13.75,100.5)` → gradient ปลอม เปลี่ยนเป็น fail loud

---

## P2 — scale / generalize / tune

- **P2-1 grid 1km/500m ไม่ใช่ grid จริง:** `grid_engine.py:434-473` subdivide 2km master ด้วย offset ตายตัว; topo `scale^0.5/^0.35` (`:707-708`) + texture `1/sqrt(ratio)` (`:721`) heuristic ล้วน; `compute_tpi_index:358-370` / `compute_local_relief:394-406` loop python ราย cell จะ OOM ที่ 276k cells
- **P2-2 tuner ไม่ represent:** `hyperparameter_tuner.py:227-229` subsample 30% + holdout เดียว แล้ว `trainer.py:97-102,337,492` reuse ข้าม resolution/provider
- **P2-3 septet dilute:** `ensemble_model.py:35-43` 7 keys + SLSQP `sum=1` (`:191,281`) weight กระจาย; `figs:60-61` / `nystroem:67-72` มี internal median impute ซ้อน; `nystroem:82` `n_components=min(120,N)` subset เล็กพัง; importance `|w|·|C|` (`:124`) เทียบกับ gain ไม่ได้แต่ถัวใน `get_cross_model_feature_attribution:408-413`
- **P2-4 cache key เปราะ:** `himawari_ir_loader.py:444-451` key จาก first/last coord + `max_dist_deg=0.05` ปัด station นอก segment ทิ้งเงียบ (`:486,569-570`)

---

## P3 — hygiene / security

- **P3-1 secret:** `openaq_loader.py:38` + `data_config.yaml:39` hardcode API key → ย้ายลง env; `grid_engine.py:67` path `E:/...` ติดมา
- **P3-2 side-effect on import:** `paths.py:220` `ensure_directories()` + `seed_*` เขียนไฟล์ตอน import
- **P3-3 RQ hardcode + save เบิ้ล:** `benchmark_suite.py:252-269` RQ2/RQ4 string ตายตัว; `ensemble_model.py:434-438` เขียน joblib เบิ้ล

---

## B. มุมนักวิจัย: ผิดหลักวิทยาศาสตร์ตรงไหน / เลขจะดีขึ้นไหม

1. **IR วัด PM ผิวดินบางๆ ในเขตร้อนชื้นแทบไม่ได้:** `BTD15-13` พิสูจน์สำหรับ elevated desert dust หนาๆ signal 1-4K แต่ PM2.5 เผาชีวมวล fine mode 0.1-0.5µm extinction ใน TIR ~0.01-0.05 จมใน diurnal 10-20K + ไอน้ำ + emissivity + noise 0.2K โมเดลจึงเรียน confounder (`hour/month/SZA/lat/elevation`) ไม่ใช่ aerosol
2. **ทฤษฎีผิดที่:** split-window สำหรับทะเลทราย/มหาสมุทร ไม่ใช่เมืองชื้นมีพืชคลุม; `B13-07` ไม่ normalize solar; `B13-10` peak ~600-700hPa ไม่ใช่ inversion ผิวดิน; `B13-11` อ้าง SO₂ ต้อง hyperspectral; `dTb/dt` ถูกแดด+เมฆกลบ; texture 6km แยก plume เมืองไม่ได้; `|∇P|` 50km IDW ได้แต่ noise ต้องการลม+PBLH; threshold 220K ดักแค่ deep convection
3. **PM2.5 vs PM10 ชุดเดียวขัด Mie:** fine << wavelength ไม่ extinct ใน TIR จะแยก 2 ตัวด้วย feature เดียวกันไม่ได้
4. **เลขจะดีบนกระดาษแต่ไม่ใช่ skill:** เพิ่ม feature static + tree ลึก + 7 ตัว + tune บน holdout เดิม MAE ต้องลงเพราะจำ `station×season×hour mean` R² 0.98 คือธง leakage ต้องชนะ 3 baseline ก่อนอ้าง: (1) `geo+time only` ไม่มีดาวเทียม (2) climatology/persistence (3) leave-province-out + night-only + high-episode ถ้า importance บนเป็น `SZA/hour/month/elevation` คือ confounded ถ้า SHAP `BTD15-13` ไม่ monotonic slope เล็ก คือไม่ได้ใช้ฟิสิกส์ฝุ่น

---

## C. Checklist ลำดับทำ

- [ ] A1 ตัด 6 time/solar ออกจาก `TOPO_SOLAR_COLUMNS` + แยก `attach_solar_meta` ไว้ eval อย่างเดียว
- [ ] A2 ตัด `coriolis_param` + เปลี่ยน target เป็น anomaly + GroupKFold/province-holdout
- [ ] P0-1/2/3/4/6 (split, imputer fit, flag, cloud, recall) — บล็อกเลขทั้งหมด
- [ ] P1 B07-correct-then-drop + รายงาน day/night + ablation ตัด B07 / เอาดาวเทียมออก
- [ ] P1 BTD alias, Tweedie/L1, HII fail-loud, pure-sat router ทะเล
- [ ] P2 grid/TPI/tuner/septet/cache ตามลำดับ
- [ ] P3 secret/path/import-side-effect/RQ
