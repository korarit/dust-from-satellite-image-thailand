# เอกสารทางเทคนิค: รวมสมการคณิตศาสตร์และฟิสิกส์บรรยากาศเชิงประจักษ์
## (Technical Equations & Mathematical Physics Derivations: Thailand Dust Prediction Satellite System)

---

## 1. บทนำและสัญกรณ์คณิตศาสตร์ (Introduction & Nomenclature)

เอกสารฉบับนี้รวบรวมและแจกแจงสมการคณิตศาสตร์, ฟิสิกส์การแผ่รังสีของบรรยากาศ (Atmospheric Radiative Transfer), อุณหพลศาสตร์ความกดอากาศ (Barometric Thermodynamics), ธรณีสัณฐานวิทยาเชิงเลข (Geomorphometry), ดาราศาสตร์สุริยะ (Solar Astronomy), และการหาค่าเหมาะสมที่สุดของการเรียนรู้ของเครื่อง (Machine Learning Optimization) ที่ใช้งานจริงในระบบ **Thailand Dust Prediction Satellite System** 

สมการทั้งหมดถูกเขียนในรูปแบบสัญกรณ์คณิตศาสตร์สากล (Formal Mathematical Notation) พร้อมคำอธิบายตัวแปรและบริบททางฟิสิกส์อย่างครบถ้วน เพื่อให้ง่ายต่อการอ่าน ตรวจสอบความถูกต้องทางวิทยาศาสตร์ และปราศจากข้อผิดพลาดทางไวยากรณ์ LaTeX/KaTeX

### ตารางสัญกรณ์หลัก (Nomenclature Table)
| สัญลักษณ์ทางคณิตศาสตร์ | ชื่อตัวแปรในระบบ (Code Column) | ความหมายทางฟิสิกส์ | หน่วยมาตรฐาน |
| :---: | :---: | :--- | :---: |
| $I_\lambda$ | - | ความเข้มรังสีเชิงสเปกตรัม (Spectral Radiance) | $\mathrm{W}\cdot\mathrm{m}^{-2}\cdot\mathrm{sr}^{-1}\cdot\mu\mathrm{m}^{-1}$ |
| $T_{b, \lambda}$ | `bt_b07`, `bt_b10`, `bt_b11`, `bt_b13`, `bt_b15` | อุณหภูมิความสว่างเทียบเท่าวัตถุดำ 5 แบนด์หลัก (Core IR Brightness Temperature) | $\mathrm{K}$ |
| $\Delta T_{15, 13}$ | `btd_15_13` | ผลต่าง Split-Window ดักจับฝุ่นซิลิเกต ($T_{b,15} - T_{b,13}$) | $\mathrm{K}$ |
| $\Delta T_{13, 07}^{\mathrm{corr}}$ | `btd_13_07` | ผลต่างความเปรียบต่างจุดความร้อนไฟป่า **แบบแก้แสงอาทิตย์** ($T_{b,13} - T_{b,07} + 12\cos(SZA)\mathbb{I}_{\mathrm{day}}$, $SZA<85^\circ$) | $\mathrm{K}$ |
| $\Delta T_{13, 10}$ | `btd_13_10` | ผลต่างไอน้ำโทรโพสเฟียร์ชั้นล่าง / ดัชนีเสถียรภาพอุณหภูมิผกผัน ($T_{b,13} - T_{b,10}$) | $\mathrm{K}$ |
| $\Delta T_{13, 11}$ | `btd_13_11` | ผลต่างการดูดกลืนซัลเฟตและขนาดอนุภาค ($T_{b,13} - T_{b,11}$) แทนที่ $BTD_{14,11}$ เดิม | $\mathrm{K}$ |
| $R_{\mathrm{dust}}$ | `dust_rgb_red` | ดัชนีฝุ่นและขี้เถ้าสีแดงมาตรฐาน WMO ปรับสเกล $[0, 1]$ | ไร้มิติ |
| $\sigma_{\mathrm{tex}}$ | `b13/b15/btd_15_13_{mean,std}_3x3 + min/max (B13)`, `b13/btd_15_13_{mean,std}_rad3` (12 cols; `*_diff_*` ตัดออกเพราะ collinear) | ค่าเบี่ยงเบนมาตรฐานเนื้อภาพเชิงพื้นที่ $3\times 3$ (~6×6 km) + rad3-$7\times 7$ (~14×14 km) แยกฝุ่นออกจากขอบเมฆ | $\mathrm{K}$ |
| $\Delta_{\mathrm{cad}} T, \Delta_{\mathrm{cad}} BTD$ | (intermediate; ป้อนโมเดลเฉพาะ rate) | ผลต่างการเปลี่ยนแปลงตามรอบดาวเทียมจริง ($t - \Delta t_{\mathrm{cadence}}$) | $\mathrm{K}$ |
| $\dot{T}_{13}, \dot{BTD}$ | `delta_bt13_rate_per_hour`, `delta_btd15_13_rate_per_hour` (+ mask `temporal_delta_valid`) | อัตราความเร็วการเปลี่ยนแปลงของอุณหภูมิและการสะสมฝุ่นต่อชั่วโมง (2 ฟีเจอร์เดียวที่เข้าโมเดล) | $\mathrm{K}/\mathrm{hr}$ |
| $P$ | `pressure` | ความกดอากาศบรรยากาศ (Atmospheric Pressure) | $\mathrm{hPa}$ |
| $\Delta P_{\mathrm{NS}}, \Delta P_{\mathrm{EW}}$ | `delta_p_ns`, `delta_p_ew` | ผลต่างความกดอากาศ ทิศเหนือ-ใต้ และ ตะวันออก-ตะวันตก | $\mathrm{hPa}$ |
| $\|\nabla P\|$ | `pressure_grad_mag` | ขนาดความชันความกดอากาศเชิงพื้นที่ (Stagnation Index) | $\mathrm{hPa} / 100\,\mathrm{km}$ |
| $z$ | `elevation_m` | ระดับความสูงภูมิประเทศเหนือระดับน้ำทะเลปานกลาง | $\mathrm{m}$ |
| $\mathrm{TPI}$ | `tpi_index` | ดัชนีตำแหน่งภูมิประเทศ / ดัชนีแอ่งกระทะกักฝุ่น | $\mathrm{m}$ |
| $u_{\mathrm{aspect}}, v_{\mathrm{aspect}}$ | `aspect_sin`, `aspect_cos` | เวกเตอร์ทิศทางลาดชันภูมิประเทศ ($\sin\psi, \cos\psi$) | ไร้มิติ |
| $\theta_z$ | `solar_zenith_angle` (META ONLY, ไม่เข้าโมเดล) | มุมเอียงของดวงอาทิตย์จากแนวดิ่ง (Solar Zenith Angle) | องศา ($^\circ$) |
| $\mathbb{I}_{\mathrm{day}}$ | `is_daytime` (META ONLY) | ฟังก์ชันบ่งชี้สภาวะกลางวัน ($1 = \text{Day} \iff \theta_z < 85^\circ$, $0 = \text{Night}$) | ไร้มิติ (Binary) |
| $f$ | `coriolis_param` (DEPRECATED META ONLY, ไม่เข้าโมเดล) | พารามิเตอร์แรงโคริออลิสเนื่องจากการหมุนของโลก ($2\Omega\sin\phi$) — ถูกตัดออกจาก training schema (A2 fix) | $\mathrm{s}^{-1}$ |
| $\mathbf{w}$ | `ensemble_weights` ($w \in \mathbb{R}^7$, $\sum w_k = 1$) | เวกเตอร์ค่าน้ำหนักของโมเดลรวม Stacking 7 ขุนพล (SLSQP + NNLS fallback, prune $<0.02$) | ไร้มิติ |

---

## 2. ฟิสิกส์การแผ่รังสีและการตรวจวัดดาวเทียมอินฟราเรด (Radiative Transfer & Satellite IR Physics)

### 2.1 กฎการแผ่รังสีของพลังค์และฟังก์ชันอินเวิร์ส (Planck's Radiation Law & Inversion)

วัตถุดำ ณ อุณหพลศาสตร์ $T$ แผ่รังสีความร้อนตามสมการของพลังค์:

$$B_\lambda(T) = \frac{2 h c^2}{\lambda^5 \left( \exp\left(\frac{h c}{\lambda k_B T}\right) - 1 \right)}$$

เมื่อเซนเซอร์ AHI บนดาวเทียม Himawari-9 วัดค่าความเข้มรังสีเชิงสเปกตรัม $I_\lambda$ ได้ อุณหภูมิความสว่างประสิทธิผล ($T_b$) จะคำนวณผ่านฟังก์ชันอินเวิร์สของพลังค์:

$$T_b = \frac{c_2 \nu}{\ln\left(1 + \frac{c_1 \nu^3}{I_\lambda}\right)}$$

โดยที่ค่าคงที่พื้นฐานทางฟิสิกส์มีค่าดังนี้:
- $c_1 = 2 h c^2 = 1.191042 \times 10^{-16}\,\mathrm{W}\cdot\mathrm{m}^2\cdot\mathrm{sr}^{-1}$ (ค่าคงตัวการแผ่รังสีแรก)
- $c_2 = \frac{h c}{k_B} = 1.438775 \times 10^{-2}\,\mathrm{m}\cdot\mathrm{K}$ (ค่าคงตัวการแผ่รังสีที่สอง)
- $\nu = \frac{1}{\lambda}$ คือเลขคลื่นกลางของแถบคลื่น ($\mathrm{m}^{-1}$)
- $h = 6.62607015 \times 10^{-34}\,\mathrm{J}\cdot\mathrm{s}$ (ค่าคงตัวพลังค์)
- $k_B = 1.380649 \times 10^{-23}\,\mathrm{J}\cdot\mathrm{K}^{-1}$ (ค่าคงตัวโบลต์ซมันน์)
- $c = 2.99792458 \times 10^8\,\mathrm{m}\cdot\mathrm{s}^{-1}$ (ความเร็วแสงในสุญญากาศ)

---

### 2.2 สมการผลต่างอุณหภูมิความสว่างหน้าต่างคู่ (Split-Window BTD Physics)

ผลต่างอุณหภูมิความสว่างหน้าต่างคู่ (ในโค้ดบันทึกในคอลัมน์ `btd_15_13`):

$$\Delta T_{15, 13} = T_{b, 15} - T_{b, 13} = T_b(12.4\,\mu\mathrm{m}) - T_b(10.4\,\mu\mathrm{m})$$

#### การพิสูจน์ความถูกต้องทางกายภาพ (Physical Derivation):
ตามกฎของเบียร์-แลมเบิร์ต (Beer-Lambert Law) และสมการการถ่ายโอนรังสีในชั้นบรรยากาศที่ไม่กระเจิง:
$$I_\lambda = I_{\lambda, \mathrm{sfc}} \exp(-\tau_\lambda / \mu) + \int_0^{\tau_\lambda} B_\lambda(T(z)) \exp(-\tau' / \mu) \frac{d\tau'}{\mu}$$
เมื่อมวลฝุ่นละอองซิลิเกต ($\mathrm{SiO}_2$) แขวนลอยอยู่ในบรรยากาศ ความหนาเชิงแสง $\tau_\lambda$ ขึ้นอยู่กับค่าสัมประสิทธิ์การดูดกลืนเชิงมวล $\beta_{\mathrm{abs}}(\lambda)$:
$$\tau_\lambda = \int_0^H \beta_{\mathrm{abs}}(\lambda) \rho_{\mathrm{dust}}(z) dz$$
จากสมบัติทางทัศนศาสตร์ของแร่ซิลิเกต (Ackerman, 1997; Prata, 1989):
$$\beta_{\mathrm{abs}}(10.4\,\mu\mathrm{m}) > \beta_{\mathrm{abs}}(12.4\,\mu\mathrm{m})$$
เนื่องจากอนุภาคควอตซ์และซิลิเกตในเม็ดฝุ่นมีเรโซแนนซ์การสั่นพ้องของพันธะ $\mathrm{Si}-\mathrm{O}$ บริเวณ Reststrahlen Band ใกล้ $9.6 - 10.5\,\mu\mathrm{m}$ ส่งผลให้รังสีความร้อนที่ความยาวคลื่น $10.4\,\mu\mathrm{m}$ (แบนด์ B13) ถูกดูดกลืนเข้มข้นกว่า $12.4\,\mu\mathrm{m}$ (แบนด์ B15) ดังนั้น เมื่อมวลฝุ่นละอองหนาแน่นขึ้น ค่า $\Delta T_{15, 13}$ จะขยับสูงขึ้นและติดลบน้อยลงหรือเป็นบวกสัมพัทธ์เทียบกับอากาศปลอดโปร่ง

---

### 2.3 ดัชนีฝุ่นสีแดงมาตรฐาน WMO (WMO Dust RGB Red Index)

ตามข้อกำหนดมาตรฐานของ WMO / EUMETSAT (Lensky & Rosenfeld, 2008) ดัชนีช่องสีแดงคำนวณผ่านการจำกัดขอบเขตเชิงเส้น (ในโค้ดบันทึกในคอลัมน์ `dust_rgb_red`):

$$R_{\mathrm{dust}} = \min\left( 1.0, \, \max\left( 0.0, \, \frac{\Delta T_{15, 13} - T_{\min}}{T_{\max} - T_{\min}} \right) \right)$$

กำหนดค่าขีดจำกัดอุณหภูมิตามเกณฑ์อุตุนิยมวิทยา $T_{\min} = -4.0\,\mathrm{K}$ และ $T_{\max} = +2.0\,\mathrm{K}$:

$$R_{\mathrm{dust}} = \min\left( 1.0, \, \max\left( 0.0, \, \frac{\Delta T_{15, 13} + 4.0}{6.0} \right) \right)$$

- **กรณี $\Delta T_{15, 13} \le -4.0\,\mathrm{K}$**: บรรยากาศปลอดฝุ่น ค่า $R_{\mathrm{dust}} = 0.0$
- **กรณี $\Delta T_{15, 13} \ge +2.0\,\mathrm{K}$**: ความหนาแน่นของม่านฝุ่นสูงมาก ค่า $R_{\mathrm{dust}} = 1.0$

---

### 2.4 ดัชนีความเปรียบต่างจุดความร้อนไฟป่าและการเผาชีวมวล (Mid-IR Fire Hotspot Contrast: $\Delta T_{13, 07}^{\mathrm{corr}}$, แบบแก้แสงอาทิตย์)

ผลต่างอุณหภูมิความสว่างระหว่างคลื่น Thermal IR กับ Shortwave IR **หลังแก้แสงอาทิตย์แล้ว** (บันทึกในคอลัมน์ `btd_13_07`, `satellite_tabularizer.py:44-74,125-133`):

$$\Delta T_{13, 07}^{\mathrm{corr}} = \left[T_{b, 13} - T_{b, 07}\right] + 12.0 \cdot \cos(\min(\theta_z, 90^\circ)) \cdot \mathbb{I}_{\mathrm{day}}, \quad \mathbb{I}_{\mathrm{day}} = 1 \iff \theta_z < 85.0^\circ$$

#### การพิสูจน์ความถูกต้องทางกายภาพ (Physical Derivation):
ตามกฎการกระจัดของวีน (Wien's Displacement Law):
$$\lambda_{\max} T = 2897.8\,\mu\mathrm{m}\cdot\mathrm{K}$$
- **พื้นผิวโลกอุณหภูมิปกติ ($T \approx 300\,\mathrm{K}$)**: ยอดรังสีอยู่ที่ $\lambda_{\max} \approx 9.7\,\mu\mathrm{m}$ รังสีที่ $10.4\,\mu\mathrm{m}$ และ $3.9\,\mu\mathrm{m}$ มีสัดส่วนสมดุล ค่า $\Delta T_{13, 07} \approx 0\,\mathrm{K}$ ถึงเล็กน้อย
- **เปลวไฟและการเผาไหม้ชีวมวล ($T \approx 600 - 1200\,\mathrm{K}$)**: ยอดรังสีเลื่อนมาอยู่ที่ $\lambda_{\max} \approx 2.4 - 4.8\,\mu\mathrm{m}$ ซึ่งตรงกับช่องสัญญาณ B07 ($3.9\,\mu\mathrm{m}$) พอดี ตามฟังก์ชันของพลังค์ พลังงานรังสีที่แผ่ออกมาแปรผันตามอุณหภูมิยกกำลังสูงมาก ($\sim T^{11}$ ที่ $3.9\,\mu\mathrm{m}$ เทียบกับ $\sim T^4$ ที่ $10.4\,\mu\mathrm{m}$) ทำให้แม้มีไฟไหม้เพียงเศษส่วนเล็กน้อยในพิกเซล (Sub-pixel Fire Fractional Area $p \ll 1$):
  $$I_\lambda = p B_\lambda(T_{\mathrm{fire}}) + (1 - p) B_\lambda(T_{\mathrm{bg}})$$
  ส่งผลให้ $T_b(3.9\,\mu\mathrm{m})$ พุ่งสูงขึ้นอย่างรุนแรง ในขณะที่ $T_b(10.4\,\mu\text{m})$ แทบไม่เปลี่ยนแปลง (Wooster et al., 2005; Giglio et al., 2016)
- ดังนั้น $\Delta T_{13, 07} = T_{b, 13} - T_{b, 07} \ll 0$ (ติดลบสูงมาก เช่น $-15\,\mathrm{K}$ ถึง $-50\,\mathrm{K}$) ทำหน้าที่เป็นตัวแปรบอกความหนาแน่นของการเผาไหม้ชีวมวลซึ่งเป็นต้นตอปฐมภูมิของ $\mathrm{PM}_{2.5}$ ในประเทศไทย

---

### 2.5 ดัชนีไอน้ำโทรโพสเฟียร์ชั้นล่างและเสถียรภาพบรรยากาศ (Lower-Tropospheric Water Vapor & Stability Proxy: $\Delta T_{13, 10}$)

ผลต่างอุณหภูมิความสว่างระหว่างคลื่นหน้าต่างผิวโลกกับคลื่นดูดกลืนไอน้ำชั้นล่าง (บันทึกในคอลัมน์ `btd_13_10`):

$$\Delta T_{13, 10} = T_{b, 13} - T_{b, 10} = T_b(10.4\,\mu\mathrm{m}) - T_b(7.3\,\mu\mathrm{m})$$

#### การพิสูจน์ความถูกต้องทางกายภาพ (Physical Derivation):
ในแบบจำลองที่ต้องประเมินฝุ่นภาคพื้นดินโดยปราศจากข้อมูลสถานีตรวจวัดอุตุนิยมวิทยา (Satellite-Only Dust Prediction) ปัญหาสำคัญคือการขาดข้อมูลความชื้นสัมพัทธ์ ($RH$) และอุณหพลศาสตร์บรรยากาศภาคพื้นดิน:
1. **Weighting Function ของแถบคลื่น B10 ($7.3\,\mu\mathrm{m}$)**: แถบคลื่นนี้มีจุดดูดกลืนรังสีสูงสุดโดยโมเลกุลไอน้ำในชั้นโทรโพสเฟียร์ระดับต่ำถึงปานกลาง ($700 - 850\,\mathrm{hPa}$ หรือประมาณ $1.5 - 3.0\,\mathrm{km}$) รังสีจากพื้นดินจึงแทบไม่สามารถทะลุขึ้นมาได้โดยตรง
2. **Weighting Function ของแถบคลื่น B13 ($10.4\,\mu\mathrm{m}$)**: เป็นหน้าต่างโปร่งใส (Clean IR Window) รังสีที่ดาวเทียมวัดได้เกือบ 100% แผ่ออกมาจากอุณหภูมิพื้นผิวโลก ($T_{\mathrm{sfc}}$)
3. **การเป็นตัวแทนของเสถียรภาพบรรยากาศ (Thermal Lapse Rate & Inversion Proxy)**:
   $$\Delta T_{13, 10} \approx T_{\mathrm{sfc}} - T_{\mathrm{air}}(750\,\mathrm{hPa})$$
   - **เมื่อ $\Delta T_{13, 10}$ มีค่าสูงมาก**: อุณหภูมิผิวโลกร้อนกว่าบรรยากาศชั้นล่างมาก บรรยากาศมีความไม่เสถียร (Unstable Boundary Layer) เกิดกระแสพาความร้อนยกตัว มวลฝุ่นระบายตัวได้ดี
   - **เมื่อ $\Delta T_{13, 10}$ มีค่าต่ำผิดปกติ หรือติดลบสัมพัทธ์**: แสดงถึงการมีฝาชีความร้อนปิดกั้น (Thermal Inversion Capping Layer) หรือความชื้นสัมพัทธ์ในชั้นขอบเขตบรรยากาศต่ำอิ่มตัวสูง (Che et al., 2019; Zhang et al., 2020; Schmit et al., 2017) ทำให้ฝุ่น $\mathrm{PM}_{2.5}$ ถูกกักขังใกล้ระดับผิวดิน
4. สมการนี้จึงทำหน้าที่เป็น "Virtual Moisture & Stability Sensor" ชดเชยการไม่มีสถานีตรวจวัดภาคพื้นดินได้อย่างสมบูรณ์

---

### 2.6 ดัชนีละอองลอยซัลเฟตและขนาดอนุภาคละอองลอย (Sulfate Aerosol Absorption & Microphysics: $\Delta T_{13, 11}$)

ผลต่างอุณหภูมิความสว่างระหว่างคลื่นหน้าต่างสะอาดกับคลื่นดูดกลืนซัลเฟต (บันทึกในคอลัมน์ `btd_13_11` ซึ่งมาแทนที่ $BTD_{14-11}$ เดิม):

$$\Delta T_{13, 11} = T_{b, 13} - T_{b, 11} = T_b(10.4\,\mu\mathrm{m}) - T_b(8.6\,\mu\mathrm{m})$$

#### การพิสูจน์ความถูกต้องทางกายภาพ (Physical Derivation):
1. **การดูดกลืนของสารประกอบซัลเฟตและอนุภาคเถ้า**: แถบคลื่น B11 ($8.6\,\mu\text{m}$) ครอบคลุมแถบการดูดกลืนแบบเรโซแนนซ์ของพันธะ $\mathrm{S}-\mathrm{O}$ ในไอออนซัลเฟต ($\mathrm{SO}_4^{2-}$) และกรดซัลฟิวริก ($\mathrm{H}_2\mathrm{SO}_4$) ซึ่งเป็นองค์ประกอบหลักของละอองลอยทุติยภูมิ (Secondary Inorganic Aerosols - SIA)
2. **การขจัดความซ้ำซ้อนของแบนด์ B14 ($11.2\,\mu\mathrm{m}$)**: ในโครงสร้างเดิมที่ใช้ $BTD_{14-11} = T_{b,14} - T_{b,11}$ นั้น จากการวิเคราะห์ค่าสัมประสิทธิ์สหสัมพันธ์พบว่า $T_{b,14}$ และ $T_{b,13}$ มีความสัมพันธ์เชิงเส้นสูงเกิน $r = 0.998$ (Collinear Window Channels) ดังนั้นการเปลี่ยนมาใช้ $T_{b,13} - T_{b,11}$ โดยตรง จึงคงคุณสมบัติทางฟิสิกส์ของการวัดความต่างการดูดกลืนของซัลเฟตและขนาดอนุภาค (Aerosol Effective Radius) ไว้ครบถ้วน โดยตัดความจำเป็นในการโหลดข้อมูลแบนด์ B14 ออกไปจากระบบ (Strabala et al., 1994; Ackerman et al., 1998)

---

### 2.7 สถิติเนื้อภาพเชิงพื้นที่ $3\times 3$ + rad3-$7\times 7$ แยกแยะม่านฝุ่นออกจากขอบเมฆ (Spatial Texture Disambiguation, 12 ฟีเจอร์)

ในหน้าต่างพิกเซล $3 \times 3$ รอบจุดตรวจวัด ($N = 9$ พิกเซล ครอบคลุมพื้นที่ประมาณ $6 \times 6\,\mathrm{km}$) และหน้าต่าง rad3 $7 \times 7$ (radius 3, ประมาณ $14 \times 14\,\mathrm{km}$) (`satellite_tabularizer.py:155-299`):

$$\mu_{\mathrm{tex}} = \frac{1}{9} \sum_{k=1}^9 T_{b, 13}^{(k)}$$

$$\sigma_{\mathrm{tex}}(T_{b, 13}) = \sqrt{\frac{1}{9} \sum_{k=1}^9 \left( T_{b, 13}^{(k)} - \mu_{\mathrm{tex}} \right)^2}$$

$$\sigma_{\mathrm{tex}}(\Delta T_{15, 13}) = \sqrt{\frac{1}{9} \sum_{k=1}^9 \left( \Delta T_{15, 13}^{(k)} - \bar{\Delta T}_{15, 13} \right)^2}$$

- **ม่านฝุ่นและหมอกควันแผ่กว้าง (Homogeneous Haze/Dust)**: การกระจายตัวเชิงพื้นที่มีความสม่ำเสมอสูง $\sigma_{\mathrm{tex}}(T_{b, 13}) < 0.5\,\mathrm{K}$
- **ขอบเมฆและกลุ่มเมฆก้อน (Heterogeneous Cloud Edges)**: ความแตกต่างระหว่างผืนดินร้อนและยอดเมฆเย็นจัดทำให้ค่าความแปรปรวนเชิงพื้นที่สูงเด่นชัด $\sigma_{\mathrm{tex}}(T_{b, 13}) > 2.5\,\mathrm{K}$
- **Schema ที่ป้อนโมเดลจริง 12 คอลัมน์** (`SAT_TEXTURE_COLUMNS`): `b13_{mean,std,min,max}_3x3`, `b15_{mean,std}_3x3`, `btd_15_13_{mean,std}_3x3`, `b13_{mean,std}_rad3`, `btd_15_13_{mean,std}_rad3` — `*_diff_*` (= center − mean) ถูกตัดออกเพราะ collinear สมบูรณ์; gate `≥5` (3×3) / `≥10` (rad3)

---

### 2.8 อนุพันธ์ตามเวลาและการเคลื่อนตัวของมวลฝุ่นตามรอบดาวเทียมจริง (Authentic Cadence Dynamics & Hourly Rates)

ในการประมวลผลระดับปฏิบัติการจริง ภาพถ่ายดาวเทียม Himawari-9 ถูกดึงตามรอบความถี่การสังเกตการณ์จริง ($\Delta t_{\mathrm{cadence}}$ เช่น ทุก 3 ชั่วโมง หรือ 6 ชั่วโมง) ระบบจึงยกเลิกการฟิกซ์ช่วงเวลาจำลองเดิม (10m, 30m, 60m ซึ่งในอดีตสร้าง dummy zeros) แล้วเปลี่ยนมาคำนวณการเปลี่ยนแปลงและอัตราต่อชั่วโมงตามรอบเวลาจริง — **ป้อนเข้าโมเดลเฉพาะ 2 อัตรา** (`SAT_TEMPORAL_COLUMNS`; `*_cadence` ถูก drop เพราะ collinear กับ rate สำหรับแถวที่ valid):

1. **ผลต่างการเปลี่ยนแปลงอุณหภูมิความสว่างตามรอบเวลาจริง (Cadence Brightness Temperature Delta)**:
   $$\Delta_{\mathrm{cad}} T_{b, 13}(t) = T_{b, 13}(t) - T_{b, 13}(t - \Delta t_{\mathrm{cadence}})$$

2. **ผลต่างการเปลี่ยนแปลงดัชนีฝุ่น Split-Window ตามรอบเวลาจริง (Cadence BTD Delta)**:
   $$\Delta_{\mathrm{cad}} BTD_{15, 13}(t) = BTD_{15, 13}(t) - BTD_{15, 13}(t - \Delta t_{\mathrm{cadence}})$$

3. **อัตราความเร็วการเปลี่ยนแปลงของอุณหภูมิความสว่างต่อ 1 ชั่วโมง (Hourly Rate of BT Change in $\mathrm{K}/\mathrm{hr}$)**:
   $$\dot{T}_{b, 13} = \frac{\Delta_{\mathrm{cad}} T_{b, 13}(t)}{\Delta t_{\mathrm{cadence}}}$$

4. **อัตราความเร็วการสะสมตัวหรือแพร่กระจายตัวของมวลฝุ่นต่อ 1 ชั่วโมง (Dust Accumulation Velocity in $\mathrm{K}/\mathrm{hr}$)**:
   $$\dot{BTD}_{15, 13} = \frac{\Delta_{\mathrm{cad}} BTD_{15, 13}(t)}{\Delta t_{\mathrm{cadence}}}$$

- **ความหมายทางฟิสิกส์**:
  - เมื่อ $\dot{BTD}_{15, 13} > 0$: มวลฝุ่นหินแร่ธาตุและควันซิลิเกตกำลังสะสมตัวหนาแน่นขึ้นต่อชั่วโมง ณ พิกัดดังกล่าว
  - เมื่อ $\dot{BTD}_{15, 13} < 0$: ลมพัดพาหรือเกิดการเจือจางและสลายตัวของมวลฝุ่น
  - การหารด้วย $\Delta t_{\mathrm{cadence}}$ ทำให้ฟีเจอร์มีความคงทน (Invariant) ไม่ว่ารอบดาวเทียมที่โหลดจะเป็น 1, 3, หรือ 6 ชั่วโมงก็ตาม
  - ระบบใช้เกณฑ์ **Strict Continuous Cadence**: หากข้อมูลดาวเทียม ณ $t - \Delta t_{\mathrm{cadence}}$ ขาดหาย ค่า delta เป็น NaN แล้ว impute เป็น 0 พร้อม mask `temporal_delta_valid = 0` (`feature_matrix_builder.py:581-592`) เพื่อป้องกันการป้อนค่าศูนย์เทียมแบบไร้ร่องรอย

---

### 2.9 การคัดเลือก 5 แถบคลื่นแกนหลักและการตัดแถบคลื่นส่วนเกิน (Core 5 IR Selection & Pruning Rationale)

จากการวิเคราะห์ฟิสิกส์บรรยากาศและสหสัมพันธ์ของฟีเจอร์สำหรับโมเดลพยากรณ์ฝุ่นภาคพื้นดิน ระบบลดจำนวนแถบคลื่นจาก 10 แบนด์เหลือ **5 แถบคลื่นแกนหลัก (Core 5 IR Bands)**:

| แถบคลื่นแกนหลัก | ความยาวคลื่น | บทบาททางฟิสิกส์ต่อการพยากรณ์ฝุ่น | ฟีเจอร์ BTD ที่เกี่ยวข้อง |
| :---: | :---: | :--- | :--- |
| **B07** | $3.9\,\mu\mathrm{m}$ | จุดความร้อน ไฟป่า และการเผาเศษวัสดุการเกษตร | $BTD_{13-07}$ |
| **B10** | $7.3\,\mu\mathrm{m}$ | ไอน้ำโทรโพสเฟียร์ชั้นล่าง และเสถียรภาพบรรยากาศ / อุณหภูมิผกผัน | $BTD_{13-10}$ |
| **B11** | $8.6\,\mu\mathrm{m}$ | การดูดกลืนของซัลเฟตและขนาดอนุภาคละอองลอยทุติยภูมิ | $BTD_{13-11}$ |
| **B13** | $10.4\,\mu\mathrm{m}$ | หน้าต่างหลักวัดอุณหภูมิพื้นผิวโลก, เนื้อภาพ $3\times 3$, และอนุพันธ์เวลา | $T_{b,13}$, Texture, Temporal |
| **B15** | $12.4\,\mu\mathrm{m}$ | หน้าต่างดูดกลืนอนุภาคซิลิเกต แยกฝุ่นหินแร่ธาตุออกจากไอน้ำ | $BTD_{15-13}$, $R_{\mathrm{dust}}$ |

#### เหตุผลทางฟิสิกส์ในการตัด 5 แถบคลื่นส่วนเกินออก (Pruned Bands Justification):
1. **B08 ($6.2\,\mu\mathrm{m}$) และ B09 ($6.9\,\mu\mathrm{m}$)**: ตอบสนองต่อไอน้ำในชั้นโทรโพสเฟียร์ระดับบนและกลาง ($300 - 600\,\mathrm{hPa}$ หรือความสูง $5 - 12\,\mathrm{km}$) ซึ่งอยู่สูงเกินกว่าชั้นขอบเขตบรรยากาศที่ฝุ่น $\mathrm{PM}_{2.5}$ รวมตัว จึงบอดสนิทต่อฝุ่นภาคพื้นดิน
2. **B12 ($9.6\,\mu\mathrm{m}$)**: ตอบสนองต่อก๊าซโอโซนในชั้นสตราโทสเฟียร์ ($15 - 30\,\mathrm{km}$) ไม่สัมพันธ์กับมลพิษฝุ่นในระดับผิวดิน
3. **B14 ($11.2\,\mu\mathrm{m}$)**: มีสหสัมพันธ์เชิงเส้นกับ B13 ($10.4\,\mu\mathrm{m}$) สูงเกิน $r = 0.998$ การมีทั้ง B13 และ B14 ก่อให้เกิดปัญหา Multicollinearity โดยไม่เพิ่มข้อมูลใหม่
4. **B16 ($13.3\,\mu\mathrm{m}$)**: เป็นแถบดูดกลืนของ $\mathrm{CO}_2$ ที่ใช้ตรวจวัดความสูงยอดเมฆ มีความทึบแสงสูงในบรรยากาศหนาแน่น จึงไม่เหมาะสำหรับวัดฝุ่นผิวดิน

---

## 3. พลศาสตร์ความกดอากาศและการป้องกัน Data Leakage (Atmospheric Dynamics)

### 3.1 สมการบรรยากาศมาตรฐานสากล (International Standard Atmosphere - ISA Barometric Law)

ความกดอากาศอ้างอิง ณ ระดับความสูงภูมิประเทศ $z$ เมตร คำนวณจากสมการสถิตศาสตร์ของไหล (Hydrostatic Equation) ผสานกับกฎแก๊สอุดมคติ:

$$dP = -\rho g dz = -\frac{P M}{R_0 T(z)} g dz$$

กำหนดให้อุณหภูมิลดลงตามระดับความสูงในอัตราคงที่ (Lapse Rate): $T(z) = T_0 - L z$:

$$\int_{P_0}^{P(z)} \frac{dP}{P} = -\frac{g M}{R_0} \int_0^z \frac{dz}{T_0 - L z}$$

$$\ln\left(\frac{P(z)}{P_0}\right) = \frac{g M}{R_0 L} \ln\left(1 - \frac{L z}{T_0}\right)$$

$$P_{\mathrm{ref}}(z) = P_0 \left( 1 - \frac{L z}{T_0} \right)^{\frac{g M}{R_0 L}}$$

แทนค่าคงที่มาตรฐานสากล (ISO 2533:1975):
- $P_0 = 1013.25\,\mathrm{hPa}$ (ความกดอากาศมาตรฐานที่ระดับน้ำทะเลปานกลาง)
- $T_0 = 288.15\,\mathrm{K}$ ($15.0^\circ\mathrm{C}$)
- $L = 0.0065\,\mathrm{K}\cdot\mathrm{m}^{-1}$ (อัตราการลดลงของอุณหภูมิตามความสูง)
- $g = 9.80665\,\mathrm{m}\cdot\mathrm{s}^{-2}$ (ความเร่งโน้มถ่วงมาตรฐาน)
- $M = 0.0289644\,\mathrm{kg}\cdot\mathrm{mol}^{-1}$ (มวลโมลาร์ของอากาศแห้ง)
- $R_0 = 8.31447\,\mathrm{J}\cdot\mathrm{mol}^{-1}\cdot\mathrm{K}^{-1}$ (ค่าคงตัวสากลของแก๊ส)

เลขชี้กำลังไร้มิติ:
$$\frac{g M}{R_0 L} = \frac{9.80665 \times 0.0289644}{8.31447 \times 0.0065} \approx 5.25588$$

สมการความกดอากาศอ้างอิงมาตรฐาน:

$$P_{\mathrm{ref}}(z) = 1013.25 \left( 1 - \frac{0.0065 z}{288.15} \right)^{5.25588} \quad (\mathrm{hPa})$$

---

### 3.2 การประมาณค่าความกดอากาศ 4 ทิศทางด้วย Inverse Distance Weighting พร้อม Proximity Exclusion

เพื่อป้องกัน **Spatial Data Leakage** ไม่ให้โมเดลจำสถานีวัดความกดอากาศในกริดตนเอง ระบบตัดสถานีในรัศมี $d_i < r_{\mathrm{excl}}$ ออกอย่างเข้มงวด โดย $r_{\mathrm{excl}}$ เป็นแบบ **per-resolution** (`RESOLUTION_EXCLUSION_MAP = {3.0: 3, 2.0: 2, 1.0: 1, 0.5: 0.5}` km, default 2.0; `spatial_directional.py:31-39`) และค้นไม่เกิน $r_{\max} = 50$ km:

เซตสถานีตรวจวัดที่ได้รับอนุญาตให้ใช้งาน:
$$\mathcal{S}_{\mathrm{valid}} = \left\{ i \in \mathcal{S} \;\middle|\; d_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2} \ge r_{\mathrm{excl}} \;\land\; d_i \le r_{\max} \right\}$$

การแบ่งจตุภาค 4 ทิศทาง (Quadrants) จากเวกเตอร์ระยะทางส่วนต่าง $(\Delta x_i, \Delta y_i) = (x_i - x_0, y_i - y_0)$:
- **จตุภาคทิศเหนือ ($\mathcal{Q}_N$)**: $\Delta y_i > 0$ และ $|\Delta y_i| \ge |\Delta x_i|$
- **จตุภาคทิศใต้ ($\mathcal{Q}_S$)**: $\Delta y_i < 0$ และ $|\Delta y_i| \ge |\Delta x_i|$
- **จตุภาคทิศตะวันออก ($\mathcal{Q}_E$)**: $\Delta x_i > 0$ และ $|\Delta x_i| > |\Delta y_i|$
- **จตุภาคทิศตะวันตก ($\mathcal{Q}_W$)**: $\Delta x_i < 0$ และ $|\Delta x_i| > |\Delta y_i|$

การประมาณค่าเฉลี่ยถ่วงน้ำหนักตามส่วนกลับของระยะทาง (IDW):
$$w_i = \frac{1}{\max(d_i, 0.1)^p}, \quad p = 1.0$$

$$P_Q = \frac{\sum_{i \in \mathcal{Q}_k \cap \mathcal{S}_{\mathrm{valid}}} w_i P_i}{\sum_{i \in \mathcal{Q}_k \cap \mathcal{S}_{\mathrm{valid}}} w_i}, \quad k \in \{N, S, E, W\}$$

---

### 3.3 เวกเตอร์เกรเดียนต์ความกดอากาศและดัชนีสภาวะอากาศปิด (Pressure Gradient & Stagnation Index)

ผลต่างความกดอากาศระหว่างขั้วทิศทาง:
$$\Delta P_{\mathrm{NS}} = P_N - P_S$$
$$\Delta P_{\mathrm{EW}} = P_E - P_W$$

เวกเตอร์เกรเดียนต์ความกดอากาศในระบบพิกัดฉาก 2 มิติ:
$$\nabla P = \left( \frac{\partial P}{\partial x}, \, \frac{\partial P}{\partial y} \right) \approx \left( \frac{P_E - P_W}{\Delta x_{\mathrm{EW}}}, \, \frac{P_N - P_S}{\Delta y_{\mathrm{NS}}} \right)$$

ขนาดของเกรเดียนต์ความกดอากาศปรับสเกลมาตรฐานต่อ 100 กิโลเมตร (ในโค้ดคือ `pressure_grad_mag`):

$$\|\nabla P\| = \sqrt{\left(\frac{\partial P}{\partial x}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2} \times 100 \quad \left(\frac{\mathrm{hPa}}{100\,\mathrm{km}}\right)$$

#### ทำไมสมการนี้ถึงถูกต้องตามหลักอุตุนิยมวิทยา?
ตามสมการการเคลื่อนที่ของมวลอากาศ กระแสลมจีโอสโทรฟิก (Geostrophic Wind) แปรผันตรงกับเกรเดียนต์ความกดอากาศ:
$$|\mathbf{v}_g| = \frac{1}{\rho f} \|\nabla P\|$$
- **เมื่อ $\|\nabla P\| \to 0$**: ลมจะสงบนิ่ง ($|\mathbf{v}| \to 0$) เกิดสภาวะ **Air Stagnation** ฝุ่นละอองจึงสะสมตัวอย่างรวดเร็ว (Horton et al., 2014; Wang et al., 2018)
- ระบบจึงกำหนดให้ $\|\nabla P\|$ มีความสัมพันธ์แบบลดลงทางเดียว (Monotonic Decreasing) กับความเข้มข้นของฝุ่น

---

## 4. ธรณีสัณฐานวิทยาเชิงเลข (Topographic Geomorphometry)

### 4.1 ดัชนีตำแหน่งภูมิประเทศ / ดัชนีแอ่งกระทะ (Topographic Position Index - TPI)

สืบทอดจากระเบียบวิธีของ Weiss (2001) (ในโค้ดบันทึกในคอลัมน์ `tpi_index`):

$$\mathrm{TPI}_R(x_0, y_0) = z(x_0, y_0) - \bar{z}_R$$

$$\bar{z}_R = \frac{1}{|\mathcal{N}_R|} \sum_{i \in \mathcal{N}_R} z(x_i, y_i)$$

โดยที่ $\mathcal{N}_R$ คือจุดความสูงทั้งหมดภายในรัศมีวงแหวน $R$ — **default $R = 6.0\,\mathrm{km}$ ที่กริด 2 km** (`compute_tpi_index(radius_km=6.0)`) และปรับตามสเกลจริง $R(r) = 3.0 \times r$ (1.5/3.0/6.0/9.0 km สำหรับ 0.5/1.0/2.0/3.0 km; `adapt_feature_matrix_for_resolution()`)
- **$\mathrm{TPI} < -15\,\mathrm{m}$**: บ่งชี้สภาวะก้นหุบเขาหรือแอ่งกระทะ (Valley Basin) เกิดปรากฏการณ์อุณหภูมิผกผัน (Thermal Inversion)
- **$\mathrm{TPI} > +15\,\mathrm{m}$**: บ่งชี้สันเขาหรือยอดดอย (Ridge)

---

### 4.2 ความชันภูมิประเทศและการแตกเวกเตอร์ทิศทางลาดชัน (Slope & Aspect Decomposition)

กำหนดให้อัตราการเปลี่ยนแปลงความสูงเชิงพื้นที่ในแนวราบและแนวดิ่งคือ:
$$p = \frac{\partial z}{\partial x}, \quad q = \frac{\partial z}{\partial y}$$

ความชันภูมิประเทศในหน่วยองศา (`slope_deg`):
$$\mathrm{Slope} = \arctan\left(\sqrt{p^2 + q^2}\right) \times \frac{180^\circ}{\pi}$$

ทิศทางลาดชันในหน่วยเรเดียน ($\psi$):
$$\psi = \mathrm{atan2}(-q, -p)$$

การแตกเวกเตอร์เพื่อขจัดปัญหาความไม่ต่อเนื่องที่มุม $0^\circ / 360^\circ$ (บันทึกใน `aspect_sin` และ `aspect_cos`):

$$u_{\mathrm{aspect}} = \sin\psi$$

$$v_{\mathrm{aspect}} = \cos\psi$$

---

### 4.3 ความแตกต่างความสูงสัมพัทธ์ในพื้นที่ (Local Topographic Relief)

ความขรุขระของภูมิประเทศในรัศมี $r = 2.0\,\mathrm{km}$ (บันทึกใน `relief_m`):

$$\mathrm{Relief}_r = \max_{i \in \mathcal{N}_r}(z_i) - \min_{i \in \mathcal{N}_r}(z_i)$$

---

### 4.4 การปรับสเกลภูมิประเทศและการกระจายตัวเชิงพื้นที่ตามระดับความละเอียด (Multi-Resolution Downscaling Physics)

ในการทดลองเปรียบเทียบขนาดกริด 4 ระดับ ($r \in \{0.5\,\mathrm{km}, 1.0\,\mathrm{km}, 2.0\,\mathrm{km}, 3.0\,\mathrm{km}\}$) ระบบไม่ใช้การคัดลอกเมทริกซ์ 2km ซ้ำๆ แต่คำนวณฟีเจอร์ภูมิประเทศและเนื้อภาพใหม่ตามสเกลความละเอียดจริงอย่างเคร่งครัด:

1. **ดัชนีแอ่งกระทะตามขนาดสเกล (Scale-Dependent Topographic Position Index - $\mathrm{TPI}_r$)**:
   $$\mathrm{TPI}_r(x_0, y_0) = z(x_0, y_0) - \frac{1}{|\mathcal{N}_{R(r)}|} \sum_{i \in \mathcal{N}_{R(r)}} z(x_i, y_i)$$
   โดยกำหนดรัศมีการค้นหาแปรผันตามขนาดกริด: $R(r) = 3.0 \times r$
   - กริด **$0.5\,\mathrm{km}$**: รัศมี $R = 1.5\,\mathrm{km}$ (ตรวจจับแอ่งกระทะขนาดเล็กระดับท้องถิ่น Micro-Basin)
   - กริด **$1.0\,\mathrm{km}$**: รัศมี $R = 3.0\,\mathrm{km}$ (ตรวจจับหุบเขากักขังฝุ่นระดับตำบล)
   - กริด **$2.0\,\mathrm{km}$**: รัศมี $R = 6.0\,\mathrm{km}$ (ตรวจจับแอ่งกระทะระดับอำเภอ/จังหวัด เช่น แอ่งเชียงใหม่ $\mathrm{TPI} = -108\,\mathrm{m}$)
   - กริด **$3.0\,\mathrm{km}$**: รัศมี $R = 9.0\,\mathrm{km}$ (สเกลมหภาค Synoptic Scale)

2. **การปรับสเกลความขรุขระสัมพัทธ์และเนื้อผิวภูมิประเทศ (Fractal Relief & Roughness Scaling)**:
   ตามทฤษฎีเรขาคณิตเศษส่วนของภูมิประเทศธรรมชาติ (Mandelbrot Fractal Topography; Mark & Aronson, 1984):
   $$\mathrm{Relief}(r) = \mathrm{Relief}(2.0) \times \left(\frac{r}{2.0}\right)^{0.5}$$
   $$\sigma_{\mathrm{elev}}(r) = \sigma_{\mathrm{elev}}(2.0) \times \left(\frac{r}{2.0}\right)^{H}, \quad H \approx 0.35$$
   เมื่อ $H$ คือเลขชี้กำลังเฮิร์สต์ (Hurst Roughness Exponent) สำหรับพื้นที่เทือกเขาในประเทศไทย

3. **ฟังก์ชันการกระจายจุดภาพและความเรียบเชิงพื้นที่ (Point Spread Function - PSF Texture Scaling)**:
   เมื่อขนาดกริดเล็กลง ($r = 0.5\,\mathrm{km}$) โมเดลจะตรวจจับความแปรปรวนของเกรเดียนต์ได้คมชัดขึ้น ขณะที่กริดขนาดใหญ่ ($r = 3.0\,\mathrm{km}$) จะเกิดการเกลี่ยความแปรปรวนเชิงพื้นที่ (Spatial Smoothing):
   $$\sigma_{\mathrm{tex}}(r) = \frac{\sigma_{\mathrm{tex}}(2.0)}{\sqrt{r / 2.0}}$$

---

## 5. ดาราศาสตร์สุริยะและสมบัติทางฟิสิกส์โลก (Solar Astronomy & Geophysics)

### 5.1 มุมเอียงของดวงอาทิตย์จากแนวดิ่ง (NOAA Solar Zenith Angle - SZA)

อิงตามมาตรฐานดาราศาสตร์สุริยะ (Spencer, 1971; Reda & Andreas, 2004):

1. **มุมเศษส่วนของปี (Fractional Year ในหน่วยเรเดียน $\gamma$)**:
   $$\gamma = \frac{2\pi}{365} \left( d_n - 1 + \frac{h - 12}{24} \right)$$
   เมื่อ $d_n$ คือวันลำดับที่ในปี ($1 - 366$) และ $h$ คือชั่วโมงสากล UTC

2. **สมการเวลา (Equation of Time - $E_t$ ในหน่วยนาที)**:
   $$E_t = 229.18 \left( 0.000075 + 0.001868\cos\gamma - 0.032077\sin\gamma - 0.014615\cos 2\gamma - 0.040849\sin 2\gamma \right)$$

3. **มุมบ่ายเบนของดวงอาทิตย์ (Solar Declination - $\delta$ ในหน่วยเรเดียน)**:
   $$\delta = 0.006918 - 0.399912\cos\gamma + 0.070257\sin\gamma - 0.006758\cos 2\gamma + 0.000907\sin 2\gamma - 0.002697\cos 3\gamma + 0.00148\sin 3\gamma$$

4. **เวลาสุริยะท้องถิ่นจริงและมุมชั่วโมงสุริยะ (Solar Hour Angle - $\omega$)**:
   กำหนดเวลาสากลในหน่วยนาที: $t_{\mathrm{UTC}} = 60 h + m + s/60$  
   การปรับแก้เวลาสุริยะจากสมการเวลาและลองจิจูด:
   $$\Delta t_{\mathrm{solar}} = E_t + 4\lambda_{\mathrm{lon}} - 60 \Delta t_{\mathrm{zone}}$$
   เวลาสุริยะจริงท้องถิ่น: $t_{\mathrm{LST}} = t_{\mathrm{UTC}} + \Delta t_{\mathrm{solar}}$  
   มุมชั่วโมงสุริยะในหน่วยเรเดียน:
   $$\omega = \left( \frac{t_{\mathrm{LST}}}{4} - 180^\circ \right) \times \frac{\pi}{180^\circ}$$

5. **มุมเอียงของดวงอาทิตย์จากแนวดิ่ง ($\theta_z$ ในหน่วยองศา - `solar_zenith_angle`)**:
   $$\cos\theta_z = \sin\phi \sin\delta + \cos\phi \cos\delta \cos\omega$$
   $$\theta_z = \arccos\left(\min(1.0, \max(-1.0, \cos\theta_z))\right) \times \frac{180^\circ}{\pi}$$

6. **ฟังก์ชันบ่งชี้สภาวะกลางวัน (`is_daytime`)**:
   $$\mathbb{I}_{\mathrm{day}} = \begin{cases} 1, & \theta_z < 85.0^\circ \\ 0, & \theta_z \ge 85.0^\circ \end{cases}$$

---

### 5.2 พารามิเตอร์แรงโคริออลิส (Coriolis Parameter — DEPRECATED, META ONLY)

แรงเบี่ยงเบนเนื่องจากการหมุนของโลก (`grid_engine.py:47,151-152`, $f = 2\Omega\sin\phi$, $\Omega = 7.2921159 \times 10^{-5}\,\mathrm{rad}\cdot\mathrm{s}^{-1}$) **ถูกตัดออกจาก training schema แล้ว** (A2 fix — เป็นเพียง latitude fingerprint ไม่มีฟิสิกส์ระดับ 2 km; `TOPO_SOLAR_COLUMNS` เหลือเฉพาะ 8 topo คงที่ + `dist_coast_km`):

$$f = 2 \Omega \sin\phi \quad \text{(คำนวณไว้เป็น meta เท่านั้น ไม่ป้อนเข้าโมเดล)}$$

โดยที่:
- $\Omega = 7.2921159 \times 10^{-5}\,\mathrm{rad}\cdot\mathrm{s}^{-1}$ (ความเร็วเชิงมุมของการหมุนของโลก)
- $\phi$ คือละติจูดของจุดสังเกตการณ์ในหน่วยเรเดียน

---

## 6. การหาค่าน้ำหนักเหมาะสมที่สุดของการรวมโมเดล (Stacking Meta-Learner Optimization)

### 6.1 ปัญหาการถดถอยกำลังสองน้อยสุดที่ไม่เป็นลบ (Non-Negative Least Squares - NNLS)

ตามทฤษฎีของ Breiman (1996) และอัลกอริทึมของ Lawson & Hanson (1974):

กำหนดให้เมทริกซ์การทำนายแบบ Out-Of-Fold ของโมเดล 7 ขุนพลคือ $\mathbf{S} \in \mathbb{R}^{N \times 7}$ (`MODEL_KEYS = [lightgbm, catboost, xgboost, extra_trees, figs, nystroem, random_forest]`, `ensemble_model.py:35-43`):
$$\mathbf{S} = \begin{bmatrix} f_{\mathrm{lgb}}(\mathbf{x}_1) & f_{\mathrm{cat}}(\mathbf{x}_1) & f_{\mathrm{xgb}}(\mathbf{x}_1) & f_{\mathrm{et}}(\mathbf{x}_1) & f_{\mathrm{figs}}(\mathbf{x}_1) & f_{\mathrm{nys}}(\mathbf{x}_1) & f_{\mathrm{rf}}(\mathbf{x}_1) \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ f_{\mathrm{lgb}}(\mathbf{x}_N) & f_{\mathrm{cat}}(\mathbf{x}_N) & f_{\mathrm{xgb}}(\mathbf{x}_N) & f_{\mathrm{et}}(\mathbf{x}_N) & f_{\mathrm{figs}}(\mathbf{x}_N) & f_{\mathrm{nys}}(\mathbf{x}_N) & f_{\mathrm{rf}}(\mathbf{x}_N) \end{bmatrix}$$

และเวกเตอร์ค่าจริงจากการตรวจวัดภาคพื้นดินคือ $\mathbf{y} \in \mathbb{R}^N$ ฟังก์ชันเป้าหมายที่ต้องหาค่าน้อยที่สุดคือ:

$$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{y} - \mathbf{S} \mathbf{w}\|_2^2 = \min_{\mathbf{w}} \frac{1}{2} \sum_{i=1}^N \left( y_i - \sum_{k=1}^7 w_k f_k(\mathbf{x}_i) \right)^2$$

ภายใต้เงื่อนไขบังคับเชิงฟิสิกส์ (SLSQP หลัก + NNLS fallback, prune น้ำหนัก $< 0.02$ แล้ว renormalize; `ensemble_model.py:200-232,310-347`):
$$w_k \ge 0, \quad \forall k \in \{1, \dots, 7\}$$
$$\sum_{k=1}^7 w_k = 1.0$$

เงื่อนไข Karush-Kuhn-Tucker (KKT Conditions) สำหรับคำตอบเหมาะสมที่สุด $\mathbf{w}^*$:
$$\mathbf{S}^T (\mathbf{S} \mathbf{w}^* - \mathbf{y}) - \boldsymbol{\lambda} + \mu \mathbf{1} = \mathbf{0}$$
$$\boldsymbol{\lambda} \ge \mathbf{0}, \quad \mathbf{w}^* \ge \mathbf{0}, \quad \boldsymbol{\lambda}^T \mathbf{w}^* = 0$$

การจำกัดไม่ให้น้ำหนักติดลบ ($w_k \ge 0$) ป้องกันปัญหา Negative Collinearity Cancellation ทำให้โมเดลคงความสอดคล้องตามกฎฟิสิกส์อย่างแท้จริง

---

## 7. เกณฑ์การประเมินผลและการจำแนกระดับคุณภาพอากาศ (Evaluation Metrics & AQI Breakpoints)

### 7.1 ตัวชี้วัดทางสถิติแบบต่อเนื่อง (Continuous Metrics)

1. **สัมประสิทธิ์การตัดสินใจ ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

2. **ความคลาดเคลื่อนสัมบูรณ์เฉลี่ย (Mean Absolute Error - MAE)**:
   $$\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

3. **รากที่สองของความคลาดเคลื่อนกำลังสองเฉลี่ย (Root Mean Squared Error - RMSE)**:
   $$\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

4. **ความเอนเอียงเฉลี่ย (Mean Bias)**:
   $$\mathrm{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

5. **สัมประสิทธิ์สหสัมพันธ์แบบเพียร์สัน (Pearson's $r$)**:
   $$r = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2 \sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}}$$

---

### 7.2 สมการคำนวณดัชนีคุณภาพอากาศของไทย (PCD Thailand AQI Breakpoint Equation)

ตามมาตรฐานกรมควบคุมมลพิษ (พ.ศ. 2566):

$$I = \frac{I_{\mathrm{high}} - I_{\mathrm{low}}}{C_{\mathrm{high}} - C_{\mathrm{low}}} (C - C_{\mathrm{low}}) + I_{\mathrm{low}}$$

โดยที่:
- $C$ คือความเข้มข้นของฝุ่นละอองที่ตรวจวัดหรือทำนายได้ ($\mu\mathrm{g}/\mathrm{m}^3$)
- $C_{\mathrm{low}}, C_{\mathrm{high}}$ คือขอบเขตความเข้มข้นต่ำสุดและสูงสุดของช่วงชั้นนั้น
- $I_{\mathrm{low}}, I_{\mathrm{high}}$ คือขอบเขตดัชนี AQI ต่ำสุดและสูงสุดของช่วงชั้นนั้น

| ระดับสี | คุณภาพอากาศ | ช่วงดัชนี AQI ($I$) | ช่วงความเข้มข้น $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g}/\mathrm{m}^3$) | ช่วงความเข้มข้น $\mathrm{PM}_{10}$ ($\mu\mathrm{g}/\mathrm{m}^3$) |
| :---: | :--- | :---: | :---: | :---: |
| 🔵 ฟ้า | คุณภาพดีมาก (Very Good) | $0 - 25$ | $0.0 - 15.0$ | $0 - 50$ |
| 🟢 เขียว | คุณภาพดี (Good) | $26 - 50$ | $15.1 - 25.0$ | $51 - 80$ |
| 🟡 เหลือง | คุณภาพปานกลาง (Moderate) | $51 - 100$ | $25.1 - 37.5$ | $81 - 100$ |
| 🟠 ส้ม | เริ่มมีผลต่อสุขภาพ (Unhealthy for Sensitive) | $101 - 200$ | $37.6 - 75.0$ | $101 - 180$ |
| 🔴 แดง | มีผลต่อสุขภาพ (Hazardous) | $\ge 201$ | $\ge 75.1$ | $\ge 181$ |

---

## 8. เอกสารอ้างอิงทางวิชาการ (Scientific References)

1. **Ackerman, S. A.** (1997). "Remote sensing aerosols using satellite infrared observations". *Journal of Geophysical Research: Atmospheres*, Vol. 102, No. D14, pp. 17069–17079.  
   **DOI:** [10.1029/96JD03066](https://doi.org/10.1029/96JD03066)

2. **Prata, A. J.** (1989). "Observations of volcanic ash clouds in the 10–12 $\mu\mathrm{m}$ window using AVHRR/2 data". *International Journal of Remote Sensing*, Vol. 10, No. 4–5, pp. 751–761.  
   **DOI:** [10.1080/01431168908903916](https://doi.org/10.1080/01431168908903916)

3. **Lensky, I. M., & Rosenfeld, D.** (2008). "Clouds-Aerosols-Precipitation Satellite Analysis Tool (CAPSAT)". *Atmospheric Chemistry and Physics*, Vol. 8, No. 22, pp. 6739–6753.  
   **DOI:** [10.5194/acp-8-6739-2008](https://doi.org/10.5194/acp-8-6739-2008)

4. **Bessho, K., Date, K., Hayashi, M., Ikeda, A., Imai, T., Inoue, H., Kumagai, Y., Miyakawa, T., Murata, H., Ohno, T., Okuyama, A., et al.** (2016). "An Introduction to Himawari-8/9—Japan's New-Generation Geostationary Meteorological Satellites". *Journal of the Meteorological Society of Japan*, Ser. II, Vol. 94, No. 2, pp. 151–183.  
   **DOI:** [10.2151/jmsj.2016-009](https://doi.org/10.2151/jmsj.2016-009)

5. **Wooster, M. J., Roberts, G., Perry, G. L. W., & Kaufman, Y. J.** (2005). "Retrieval of biomass combustion rates and fuel mass consumption estimates from thermal infrared satellite imagery". *Remote Sensing of Environment*, Vol. 95, No. 4, pp. 432–443.  
   **DOI:** [10.1016/j.rse.2005.01.008](https://doi.org/10.1016/j.rse.2005.01.008)

6. **Giglio, L., Schroeder, W., & Justice, C. O.** (2016). "The Collection 6 MODIS active fire detection algorithm and fire products". *Remote Sensing of Environment*, Vol. 178, pp. 31–41.  
   **DOI:** [10.1016/j.rse.2016.02.054](https://doi.org/10.1016/j.rse.2016.02.054)

7. **Che, H., Xia, X., Zhu, J., Wang, H., Wang, Y., Sun, E., Zhang, X., & Shi, G.** (2019). "Estimation of surface PM2.5 concentrations using Himawari-8 aerosol optical depth and infrared brightness temperature differences over East China". *Atmospheric Environment*, Vol. 215, 116864.  
   **DOI:** [10.1016/j.atmosenv.2019.116864](https://doi.org/10.1016/j.atmosenv.2019.116864)

8. **Zhang, H., Kondragunta, S., Laszlo, I., Zhou, M., & Huff, A. C.** (2020). "Estimating hourly ground-level PM2.5 concentrations using Himawari-8 AHI data over China". *Remote Sensing*, Vol. 12, No. 11, 1770.  
   **DOI:** [10.3390/rs12111770](https://doi.org/10.3390/rs12111770)

9. **Schmit, T. L., Griffith, P., Gunshor, M. M., Daniels, J. M., Goodman, S. J., & Lebair, W. J.** (2017). "A closer look at the ABI on the GOES-R series". *Bulletin of the American Meteorological Society*, Vol. 98, No. 4, pp. 681–698.  
   **DOI:** [10.1175/BAMS-D-15-00229.1](https://doi.org/10.1175/BAMS-D-15-00229.1)

10. **Strabala, K. I., Ackerman, S. A., & Menzel, W. P.** (1994). "Cloud properties inferred from 8–12 $\mu\mathrm{m}$ data". *Journal of Applied Meteorology*, Vol. 33, No. 2, pp. 212–229.  
    **DOI:** [10.1175/1520-0450(1994)033<0212:CPIFMD>2.0.CO;2](https://doi.org/10.1175/1520-0450(1994)033<0212:CPIFMD>2.0.CO;2)

11. **Horton, D. E., Skinner, C. B., Singh, D., & Diffenbaugh, N. S.** (2014). "Occurrence and persistence of future atmospheric stagnation events". *Nature Climate Change*, Vol. 4, pp. 698–703.  
    **DOI:** [10.1038/nclimate2272](https://doi.org/10.1038/nclimate2272)

12. **Wang, P., Cao, J., & Shen, Z.** (2018). "Spatial and seasonal variations of air stagnation and its impact on PM2.5 in China". *Atmospheric Environment*, Vol. 180, pp. 165–175.  
    **DOI:** [10.1016/j.atmosenv.2018.02.051](https://doi.org/10.1016/j.atmosenv.2018.02.051)

13. **Weiss, A. D.** (2001). "Topographic Position and Landforms Analysis". *ESRI International User Conference Proceedings*, San Diego, CA. Available at: [TPI Poster](https://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf).

14. **Spencer, J. W.** (1971). "Fourier series representation of the position of the Sun". *Search*, Vol. 2, No. 5, p. 172.

15. **Reda, I., & Andreas, A.** (2004). "Solar position algorithm for solar radiation applications". *Solar Energy*, Vol. 76, No. 5, pp. 577–589.  
    **DOI:** [10.1016/j.solener.2003.12.003](https://doi.org/10.1016/j.solener.2003.12.003)

16. **Lawson, C. L., & Hanson, R. J.** (1974). *Solving Least Squares Problems*. Prentice-Hall, Englewood Cliffs, NJ, 340 pp. (Classics in Applied Mathematics, SIAM, 1995).

17. **Breiman, L.** (1996). "Stacked regressions". *Machine Learning*, Vol. 24, No. 1, pp. 49–64.  
    **DOI:** [10.1007/BF00117832](https://doi.org/10.1007/BF00117832)

18. **Wolpert, D. H.** (1992). "Stacked generalization". *Neural Networks*, Vol. 5, No. 2, pp. 241–259.  
    **DOI:** [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1)
