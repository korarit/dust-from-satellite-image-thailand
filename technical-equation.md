# เอกสารทางเทคนิค: รวมสมการคณิตศาสตร์และฟิสิกส์บรรยากาศเชิงประจักษ์
## (Technical Equations & Mathematical Physics Derivations: Thailand Dust Prediction Satellite System)

---

## 1. บทนำและสัญกรณ์คณิตศาสตร์ (Introduction & Nomenclature)

เอกสารฉบับนี้รวบรวมและแจกแจงสมการคณิตศาสตร์, ฟิสิกส์การแผ่รังสีของบรรยากาศ (Atmospheric Radiative Transfer), อุณหพลศาสตร์ความกดอากาศ (Barometric Thermodynamics), ธรณีสัณฐานวิทยาเชิงเลข (Geomorphometry), ดาราศาสตร์สุริยะ (Solar Astronomy), และการหาค่าเหมาะสมที่สุดของการเรียนรู้ของเครื่อง (Machine Learning Optimization) ที่ถูกนำมาใช้งานจริงในระบบ **Thailand Dust Prediction Satellite** ทุกสมการมีที่มาทางทฤษฎีรองรับ มีความถูกต้องทางกายภาพ และปราศจากการสุ่มหรือการประมาณค่าเทียม (Strict Zero-Mock)

### ตารางสัญกรณ์หลัก (Key Nomenclature)
| สัญลักษณ์ | ความหมาย | หน่วยมาตรฐาน SI |
| :---: | :--- | :---: |
| $I_\lambda$ | ความเข้มรังสีเชิงสเปกตรัม (Spectral Radiance) | $\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}$ |
| $T_b$ | อุณหภูมิความสว่างเทียบเท่าวัตถุดำ (Brightness Temperature) | เคลวิน ($\text{K}$) |
| $BTD_{ij}$ | ผลต่างอุณหภูมิความสว่างระหว่างแบนด์ $i$ และ $j$ ($T_{b,i} - T_{b,j}$) | เคลวิน ($\text{K}$) |
| $P$ | ความกดอากาศบรรยากาศ (Atmospheric Pressure) | เฮกโตปาสกาล ($\text{hPa}$) |
| $\nabla P$ | เวกเตอร์ความชันความกดอากาศเชิงพื้นที่ (Pressure Gradient Vector) | $\text{hPa} / 100\,\text{km}$ |
| $\|\nabla P\|$ | ขนาดของความชันความกดอากาศ (ความรุนแรงของสภาวะอากาศปิด) | $\text{hPa} / 100\,\text{km}$ |
| $z$ | ระดับความสูงภูมิประเทศเหนือระดับน้ำทะเลปานกลาง (Elevation) | เมตร ($\text{m}$) |
| $TPI$ | ดัชนีตำแหน่งภูมิประเทศ / ดัชนีแอ่งกระทะ (Topographic Position Index) | เมตร ($\text{m}$) |
| $\theta_z$ | มุมเอียงของดวงอาทิตย์จากแนวดิ่ง (Solar Zenith Angle - SZA) | องศา ($^\circ$) |
| $f$ | พารามิเตอร์โคริออลิส (Coriolis Parameter) | $\text{s}^{-1}$ |
| $\mathbf{w}$ | เวกเตอร์ค่าน้ำหนักของแบบจำลองรวม (Stacking Meta-Learner Weights) | ไร้มิติ ($\sum w_i = 1$) |

---

## 2. ฟิสิกส์การแผ่รังสีและการตรวจวัดดาวเทียมอินฟราเรด (Radiative Transfer & Satellite IR Physics)

### 2.1 กฎการแผ่รังสีของพลังค์และฟังก์ชันอินเวิร์ส (Planck's Law & Inversion)
วัตถุดำ ณ อุณหภูมิอุณหพลศาสตร์ $T$ จะแผ่รังสีความร้อนตามสมการของพลังค์:

$$B_\lambda(T) = \frac{2 h c^2}{\lambda^5 \left( \exp\left(\frac{h c}{\lambda k_B T}\right) - 1 \right)}$$

เมื่อเซนเซอร์ AHI บนดาวเทียม Himawari-9 วัดค่าความเข้มรังสีเชิงสเปกตรัม $I_\lambda$ ได้ อุณหภูมิความสว่างประสิทธิผล ($T_b$) จะคำนวณผ่านฟังก์ชันอินเวิร์ส:

$$T_b = \frac{c_2 \nu}{\ln\left(1 + \frac{c_1 \nu^3}{I_\lambda}\right)}$$

โดยที่:
- $c_1 = 2 h c^2 = 1.191042 \times 10^{-16}\,\text{W}\cdot\text{m}^2\cdot\text{sr}^{-1}$
- $c_2 = \frac{h c}{k_B} = 1.438775 \times 10^{-2}\,\text{m}\cdot\text{K}$
- $\nu = \frac{1}{\lambda}$ คือเลขคลื่น ($\text{m}^{-1}$)
- $h = 6.62607015 \times 10^{-34}\,\text{J}\cdot\text{s}$ (ค่าคงตัวพลังค์)
- $k_B = 1.380649 \times 10^{-23}\,\text{J}\cdot\text{K}^{-1}$ (ค่าคงตัวโบลต์ซมันน์)
- $c = 2.99792458 \times 10^8\,\text{m}\cdot\text{s}^{-1}$ (ความเร็วแสงในสุญญากาศ)

---

### 2.2 สมการผลต่างอุณหภูมิความสว่างหน้าต่างคู่ (Split-Window BTD Physics)

$$BTD_{15-13} = T_b(12.4\,\mu\text{m}) - T_b(10.4\,\mu\text{m})$$

#### การพิสูจน์ความถูกต้องทางกายภาพ (Physical Derivation):
ตามกฎของเบียร์-แลมเบิร์ต (Beer-Lambert Law) และสมการการถ่ายโอนรังสีในชั้นบรรยากาศที่ไม่กระเจิง (Non-scattering Atmosphere):
$$I_\lambda = I_{\lambda, \text{sfc}} \exp(-\tau_\lambda / \mu) + \int_0^{\tau_\lambda} B_\lambda(T(z)) \exp(-\tau' / \mu) \frac{d\tau'}{\mu}$$
เมื่อมวลฝุ่นละอองซิลิเกต ($\text{SiO}_2$) แขวนลอยอยู่ในบรรยากาศ ความหนาเชิงแสง $\tau_\lambda$ ขึ้นอยู่กับค่าสัมประสิทธิ์การดูดกลืนเชิงมวล $\beta_{\text{abs}}(\lambda)$:
$$\tau_\lambda = \int_0^H \beta_{\text{abs}}(\lambda) \rho_{\text{dust}}(z) dz$$
จากสมบัติเชิงแสงของแร่ซิลิเกต (Ackerman, 1997; Prata, 1989):
$$\beta_{\text{abs}}(10.4\,\mu\text{m}) > \beta_{\text{abs}}(12.4\,\mu\text{m})$$
เนื่องจากอนุภาคควอตซ์และซิลิเกตมีการสั่นพ้องของพันธะ $\text{Si}-\text{O}$ บริเวณ Reststrahlen Band ใกล้ $9.6 - 10.5\,\mu\text{m}$ ทำให้รังสีที่ความยาวคลื่น $10.4\,\mu\text{m}$ (แบนด์ B13) ถูกดูดกลืนมากกว่า $12.4\,\mu\text{m}$ (แบนด์ B15) ผลต่าง $BTD_{15-13} = T_{b,15} - T_{b,13}$ จึงมีค่าลดลงและติดลบน้อยลงหรือเป็นบวกสัมพัทธ์เมื่อเทียบกับบรรยากาศที่ปราศจากฝุ่น ทำให้สมการนี้มีความถูกต้องตามหลักการทางทัศนศาสตร์ของอนุภาคอย่างแท้จริง

---

### 2.3 ดัชนีฝุ่นและขี้เถ้าสีแดงมาตรฐาน WMO (WMO Dust RGB Red Index)

ตามข้อกำหนดของ WMO / EUMETSAT (Lensky & Rosenfeld, 2008):

$$I_{\text{dust, red}} = \text{clip}\left( \frac{BTD_{15-13} - T_{\min}}{T_{\max} - T_{\min}}, \, 0.0, \, 1.0 \right)$$

โดยที่ $T_{\min} = -4.0\,\text{K}$ และ $T_{\max} = +2.0\,\text{K}$:

$$I_{\text{dust, red}} = \max\left( 0.0, \, \min\left( 1.0, \, \frac{BTD_{15-13} + 4.0}{6.0} \right) \right)$$

- **เมื่อ $BTD_{15-13} \le -4.0\,\text{K}$**: ชั้นบรรยากาศชื้น ไร้ฝุ่นละออง ค่าดัชนีเท่ากับ $0.0$
- **เมื่อ $BTD_{15-13} \ge +2.0\,\text{K}$**: ความหนาแน่นของฝุ่นและหมอกควันสูงมาก ค่าดัชนีอิ่มตัวที่ $1.0$

---

## 3. พลศาสตร์ความกดอากาศและการป้องกัน Data Leakage (Atmospheric Dynamics)

### 3.1 สมการบรรยากาศมาตรฐานสากล (International Standard Atmosphere - ISA Barometric Law)
ความกดอากาศอ้างอิง ณ ระดับความสูงภูมิประเทศ $z$ เมตร คำนวณจากสมการสถิตศาสตร์ของไหล (Hydrostatic Equation) ผสานกับกฎของแก๊สอุดมคติ:

$$dP = -\rho g dz = -\frac{P M}{R_0 T(z)} g dz$$

กำหนดให้อุณหภูมิลดลงตามระดับความสูงในอัตราคงที่ (Lapse Rate): $T(z) = T_0 - L z$:

$$\int_{P_0}^{P(z)} \frac{dP}{P} = -\frac{g M}{R_0} \int_0^z \frac{dz}{T_0 - L z}$$

$$\ln\left(\frac{P(z)}{P_0}\right) = \frac{g M}{R_0 L} \ln\left(1 - \frac{L z}{T_0}\right)$$

$$P_{\text{ref}}(z) = P_0 \left( 1 - \frac{L z}{T_0} \right)^{\frac{g M}{R_0 L}}$$

แทนค่าคงที่มาตรฐานสากล (ISO 2533:1975):
- $P_0 = 1013.25\,\text{hPa}$ (ความกดอากาศมาตรฐานที่ระดับน้ำทะเล)
- $T_0 = 288.15\,\text{K}$ ($15.0^\circ\text{C}$)
- $L = 0.0065\,\text{K}\cdot\text{m}^{-1}$ (อัตราการลดลงของอุณหภูมิ $6.5\,\text{K}/\text{km}$)
- $g = 9.80665\,\text{m}\cdot\text{s}^{-2}$ (ความเร่งโน้มถ่วงมาตรฐาน)
- $M = 0.0289644\,\text{kg}\cdot\text{mol}^{-1}$ (มวลโมลาร์ของอากาศแห้ง)
- $R_0 = 8.31447\,\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$ (ค่าคงตัวสากลของแก๊ส)

เลขชี้กำลังไร้มิติ:
$$\frac{g M}{R_0 L} = \frac{9.80665 \times 0.0289644}{8.31447 \times 0.0065} \approx 5.25588$$

ดังนั้น สมการความกดอากาศอ้างอิงที่ใช้งานจริงคือ:

$$P_{\text{ref}}(z) = 1013.25 \left( 1 - \frac{0.0065 z}{288.15} \right)^{5.25588} \quad (\text{hPa})$$

---

### 3.2 การประมาณค่าความกดอากาศ 4 ทิศทางด้วย Inverse Distance Weighting พร้อม Proximity Exclusion

เพื่อป้องกันปัญหา **Spatial Data Leakage** ไม่ให้โมเดลจำค่าความกดอากาศของสถานีที่ตั้งอยู่ในกริดตัวเอง ระบบกำหนดเงื่อนไขการตัดสถานีในรัศมี $r < r_{\text{excl}}$ (เช่น $r_{\text{excl}} = 2.0\,\text{km}$ สำหรับกริด 2km) ออกอย่างเด็ดขาด:

สถานีที่นำมาคำนวณจะต้องสอดคล้องกับ:
$$\mathcal{S}_{\text{valid}} = \left\{ i \in \mathcal{S} \;\middle|\; d_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2} \ge r_{\text{excl}} \;\land\; d_i \le r_{\max} \right\}$$

การแบ่งจตุภาค 4 ทิศทาง (Quadrants) จากเวกเตอร์ส่วนต่าง $(\Delta x_i, \Delta y_i) = (x_i - x_0, y_i - y_0)$:
- **ทิศเหนือ ($\mathcal{Q}_N$)**: $\Delta y_i > 0$ และ $|\Delta y_i| \ge |\Delta x_i|$
- **ทิศใต้ ($\mathcal{Q}_S$)**: $\Delta y_i < 0$ และ $|\Delta y_i| \ge |\Delta x_i|$
- **ทิศตะวันออก ($\mathcal{Q}_E$)**: $\Delta x_i > 0$ และ $|\Delta x_i| > |\Delta y_i|$
- **ทิศตะวันตก ($\mathcal{Q}_W$)**: $\Delta x_i < 0$ และ $|\Delta x_i| > |\Delta y_i|$

การถ่วงน้ำหนักตามส่วนกลับของระยะทาง (IDW):
$$w_i = \frac{1}{\max(d_i, 0.1)^p}, \quad p = 1.0$$

$$P_Q = \frac{\sum_{i \in \mathcal{Q}_k} w_i P_i}{\sum_{i \in \mathcal{Q}_k} w_i}, \quad k \in \{N, S, E, W\}$$

---

### 3.3 เวกเตอร์เกรเดียนต์ความกดอากาศและดัชนีสภาวะอากาศปิด (Pressure Gradient & Stagnation Index)

ความแตกต่างของความกดอากาศระหว่างทิศเหนือ-ใต้ และ ตะวันออก-ตะวันตก:
$$\Delta P_{NS} = P_N - P_S$$
$$\Delta P_{EW} = P_E - P_W$$

เวกเตอร์เกรเดียนต์ความกดอากาศใน 2 มิติ:
$$\nabla P = \left( \frac{\partial P}{\partial x}, \, \frac{\partial P}{\partial y} \right) \approx \left( \frac{P_E - P_W}{\Delta x_{EW}}, \, \frac{P_N - P_S}{\Delta y_{NS}} \right)$$

ขนาดของเกรเดียนต์ความกดอากาศต่อระยะทาง 100 กิโลเมตร ($\|\nabla P\|$):
$$\|\nabla P\| = \sqrt{\left(\frac{\partial P}{\partial x}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2} \times 100 \quad \left(\frac{\text{hPa}}{100\,\text{km}}\right)$$

#### ทำไมสมการนี้ถึงถูกต้องตามหลักอุตุนิยมวิทยา?
ตามสมการการเคลื่อนที่ของมวลอากาศ (Equation of Atmospheric Motion) กระแสลมจีโอสโทรฟิก (Geostrophic Wind) สัมพันธ์โดยตรงกับความชันความกดอากาศ:
$$u_g = -\frac{1}{\rho f} \frac{\partial P}{\partial y}, \quad v_g = \frac{1}{\rho f} \frac{\partial P}{\partial x}$$
$$|\mathbf{v}_g| = \frac{1}{\rho f} \|\nabla P\|$$
- **เมื่อ $\|\nabla P\| \to 0$**: ความชันความกดอากาศเข้าใกล้ศูนย์ กระแสลมจะสงบนิ่ง ($|\mathbf{v}| \to 0$) เกิดสภาวะ **Air Stagnation** มวลอากาศไม่สามารถระบายตัวออกสู่ภายนอกได้ ฝุ่นละออง $\text{PM}_{2.5}$ จึงสะสมตัวอย่างรวดเร็ว (Horton et al., 2014; Wang et al., 2018)
- ระบบจึงกำหนดให้ $\|\nabla P\|$ มีความสัมพันธ์แบบลดลงทางเดียว (Monotonic Decreasing Constraint) กับความเข้มข้นของฝุ่น

---

## 4. ธรณีสัณฐานวิทยาเชิงเลข (Topographic Geomorphometry)

### 4.1 ดัชนีตำแหน่งภูมิประเทศ / ดัชนีแอ่งกระทะ (Topographic Position Index - TPI)
สืบทอดจากระเบียบวิธีของ Weiss (2001):

$$TPI_R(x_0, y_0) = z(x_0, y_0) - \bar{z}_R$$

$$\bar{z}_R = \frac{1}{N_R} \sum_{i \in \mathcal{N}_R} z(x_i, y_i)$$

โดยที่ $\mathcal{N}_R$ คือจุดความสูงทั้งหมดภายในรัศมีวงแหวน $R = 6.0\,\text{km}$
- **$TPI < -15\,\text{m}$**: จัดเป็นร่องเขาหรือก้นหุบเขา (Basin/Valley Bottom) เกิดการสะสมของมวลอากาศเย็นที่จมตัวลง (Cold Pool)
- **$TPI > +15\,\text{m}$**: จัดเป็นสันเขาหรือยอดดอย (Ridge)

---

### 4.2 ความชันภูมิประเทศและการแตกเวกเตอร์ทิศทางลาดชัน (Slope & Aspect Decomposition)

กำหนดให้ระดับความสูง $z(x, y)$ มีอัตราการเปลี่ยนแปลงเชิงพื้นที่:
$$p = \frac{\partial z}{\partial x}, \quad q = \frac{\partial z}{\partial y}$$

$$\text{Slope} = \arctan\left(\sqrt{p^2 + q^2}\right) \times \frac{180^\circ}{\pi} \quad (\text{degrees})$$

$$\text{Aspect} = \text{atan2}(-q, -p) \quad (\text{radians})$$

เพื่อขจัดปัญหาความไม่ต่อเนื่องเชิงมุม (Discontinuity at $0^\circ / 360^\circ$):
$$\text{aspect\_sin} = \sin(\text{Aspect})$$
$$\text{aspect\_cos} = \cos(\text{Aspect})$$

---

### 4.3 ความแตกต่างความสูงสัมพัทธ์ในพื้นที่ (Local Topographic Relief)
$$\text{Relief}_r = \max_{i \in \mathcal{N}_r}(z_i) - \min_{i \in \mathcal{N}_r}(z_i)$$
วัดความขรุขระของผืนดินและการกักกั้นการไหลเวียนของลมระดับผิวพื้น

---

## 5. ดาราศาสตร์สุริยะและสมบัติทางฟิสิกส์โลก (Solar Astronomy & Geophysics)

### 5.1 มุมเอียงของดวงอาทิตย์จากแนวดิ่ง (NOAA Solar Zenith Angle - SZA)
อิงตามอัลกอริทึมดาราศาสตร์สุริยะสากล (Spencer, 1971; Reda & Andreas, 2004):

มุมเศษส่วนของปี (Fractional Year ในหน่วยเรเดียน):
$$\gamma = \frac{2\pi}{365} \left( d_n - 1 + \frac{h - 12}{24} \right)$$
เมื่อ $d_n$ คือวันลำดับที่ในปี (Day of the year, $1 - 366$) และ $h$ คือชั่วโมง UTC

สมการเวลา (Equation of Time - $E_t$ ในหน่วยนาที):
$$E_t = 229.18 \left( 0.000075 + 0.001868\cos\gamma - 0.032077\sin\gamma - 0.014615\cos 2\gamma - 0.040849\sin 2\gamma \right)$$

มุมบ่ายเบนของดวงอาทิตย์ (Solar Declination - $\delta$ ในหน่วยเรเดียน):
$$\delta = 0.006918 - 0.399912\cos\gamma + 0.070257\sin\gamma - 0.006758\cos 2\gamma + 0.000907\sin 2\gamma - 0.002697\cos 3\gamma + 0.00148\sin 3\gamma$$

มุมชั่วโมงสุริยะจริง (Solar Hour Angle - $\omega$):
$$\text{Time Offset} = E_t + 4 \times \lambda_{\text{lon}} - 60 \times \text{Timezone}$$
$$\text{True Solar Time (min)} = h \times 60 + m + \frac{s}{60} + \text{Time Offset}$$
$$\omega = \left( \frac{\text{True Solar Time}}{4} - 180^\circ \right) \times \frac{\pi}{180^\circ}$$

มุมเอียงของดวงอาทิตย์จากแนวดิ่ง ($\theta_z$):
$$\cos\theta_z = \sin\phi \sin\delta + \cos\phi \cos\delta \cos\omega$$
$$\theta_z = \arccos\left(\text{clip}(\cos\theta_z, -1.0, 1.0)\right) \times \frac{180^\circ}{\pi}$$

เงื่อนไขการจำแนกกลางวัน/กลางคืน (Day/Night Indicator):
$$\text{is\_daytime} = \begin{cases} 1, & \theta_z < 85.0^\circ \\ 0, & \theta_z \ge 85.0^\circ \end{cases}$$

---

### 5.2 พารามิเตอร์แรงโคริออลิส (Coriolis Parameter)
แรงเบี่ยงเบนเนื่องจากการหมุนรอบตัวเองของโลกส่งผลต่อการก่อตัวของลมกรดและความเสถียรของชั้นบรรยากาศ:
$$f = 2 \Omega \sin\phi$$
โดยที่:
- $\Omega = 7.2921159 \times 10^{-5}\,\text{rad}\cdot\text{s}^{-1}$ (ความเร็วเชิงมุมในการหมุนของโลก)
- $\phi$ คือละติจูดของจุดสังเกตการณ์ในหน่วยเรเดียน

---

## 6. การหาค่าน้ำหนักเหมาะสมที่สุดของการรวมโมเดล (Stacking Meta-Learner Optimization)

### 6.1 ปัญหาการถดถอยกำลังสองน้อยสุดที่ไม่เป็นลบ (Non-Negative Least Squares - NNLS)
ตามทฤษฎีของ Breiman (1996) และอัลกอริทึมของ Lawson & Hanson (1974) การรวมโมเดล 4 ขุนพล (LightGBM, CatBoost, XGBoost, Random Forest) ต้องไม่ยอมให้ค่าน้ำหนักติดลบ เพื่อป้องกันไม่ให้เกิด Negative Feature Cancellation ที่ขัดแย้งกับหลักฟิสิกส์:

กำหนดให้เมทริกซ์การทำนายแบบ Out-Of-Fold คือ $\mathbf{S} \in \mathbb{R}^{N \times K}$ (โดย $K = 4$):
$$\mathbf{S} = \begin{bmatrix} f_{\text{lgb}}(\mathbf{x}_1) & f_{\text{cat}}(\mathbf{x}_1) & f_{\text{xgb}}(\mathbf{x}_1) & f_{\text{rf}}(\mathbf{x}_1) \\ \vdots & \vdots & \vdots & \vdots \\ f_{\text{lgb}}(\mathbf{x}_N) & f_{\text{cat}}(\mathbf{x}_N) & f_{\text{xgb}}(\mathbf{x}_N) & f_{\text{rf}}(\mathbf{x}_N) \end{bmatrix}$$

และเวกเตอร์ค่าจริงคือ $\mathbf{y} \in \mathbb{R}^N$ ฟังก์ชันเป้าหมายที่ต้องหาค่าน้อยที่สุดคือ:

$$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{y} - \mathbf{S} \mathbf{w}\|_2^2 = \min_{\mathbf{w}} \frac{1}{2} \sum_{i=1}^N \left( y_i - \sum_{k=1}^K w_k f_k(\mathbf{x}_i) \right)^2$$

ภายใต้เงื่อนไขบังคับ (Constraints):
$$w_k \ge 0, \quad \forall k \in \{1, \dots, K\}$$
$$\sum_{k=1}^K w_k = 1.0$$

เงื่อนไข Karush-Kuhn-Tucker (KKT Conditions) สำหรับคำตอบเหมาะสมที่สุด $\mathbf{w}^*$:
$$\mathbf{S}^T (\mathbf{S} \mathbf{w}^* - \mathbf{y}) - \boldsymbol{\lambda} + \mu \mathbf{1} = \mathbf{0}$$
$$\boldsymbol{\lambda} \ge \mathbf{0}, \quad \mathbf{w}^* \ge \mathbf{0}, \quad \boldsymbol{\lambda}^T \mathbf{w}^* = 0$$

ผลลัพธ์ทำให้โมเดลรวมรักษาสมดุลความแม่นยำสูงสุดโดยมีค่าความคลาดเคลื่อนต่ำกว่าโมเดลเดี่ยวใดๆ เสมอ

---

## 7. เกณฑ์การประเมินผลและการจำแนกระดับคุณภาพอากาศ (Evaluation Metrics & AQI Breakpoints)

### 7.1 ตัวชี้วัดทางสถิติแบบต่อเนื่อง (Continuous Metrics)
1. **สัมประสิทธิ์การตัดสินใจ ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
2. **ความคลาดเคลื่อนสัมบูรณ์เฉลี่ย (Mean Absolute Error - MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **รากที่สองของความคลาดเคลื่อนกำลังสองเฉลี่ย (Root Mean Squared Error - RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
4. **ความเอนเอียงเฉลี่ย (Mean Bias)**:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
5. **สัมประสิทธิ์สหสัมพันธ์แบบเพียร์สัน (Pearson's $r$)**:
   $$r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}$$

---

### 7.2 สมการคำนวณดัชนีคุณภาพอากาศของไทย (PCD Thailand AQI Formula)
ตามมาตรฐานกรมควบคุมมลพิษ (พ.ศ. 2566):

$$I = \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} (C - C_{\text{low}}) + I_{\text{low}}$$

| ระดับสี | ระดับคุณภาพอากาศ | ช่วงดัชนี AQI ($I$) | ช่วงความเข้มข้น $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) | ช่วงความเข้มข้น $\text{PM}_{10}$ ($\mu\text{g/m}^3$) |
| :---: | :--- | :---: | :---: | :---: |
| 🔵 ฟ้า | คุณภาพอากาศดีมาก (Very Good) | $0 - 25$ | $0.0 - 15.0$ | $0 - 50$ |
| 🟢 เขียว | คุณภาพอากาศดี (Good) | $26 - 50$ | $15.1 - 25.0$ | $51 - 80$ |
| 🟡 เหลือง | คุณภาพปานกลาง (Moderate) | $51 - 100$ | $25.1 - 37.5$ | $81 - 100$ |
| 🟠 ส้ม | เริ่มมีผลกระทบต่อสุขภาพ (Unhealthy for Sensitive) | $101 - 200$ | $37.6 - 75.0$ | $101 - 180$ |
| 🔴 แดง | มีผลกระทบต่อสุขภาพ (Hazardous) | $\ge 201$ | $\ge 75.1$ | $\ge 181$ |

---

## 8. เอกสารอ้างอิงทางวิชาการ (Scientific References)

1. **Ackerman, S. A.** (1997). "Remote sensing aerosols using satellite infrared observations". *Journal of Geophysical Research: Atmospheres*, Vol. 102, No. D14, pp. 17069–17079.  
   **DOI:** [10.1029/96JD03066](https://doi.org/10.1029/96JD03066)

2. **Prata, A. J.** (1989). "Observations of volcanic ash clouds in the 10–12 $\mu\text{m}$ window using AVHRR/2 data". *International Journal of Remote Sensing*, Vol. 10, No. 4–5, pp. 751–761.  
   **DOI:** [10.1080/01431168908903916](https://doi.org/10.1080/01431168908903916)

3. **Lensky, I. M., & Rosenfeld, D.** (2008). "Clouds-Aerosols-Precipitation Satellite Analysis Tool (CAPSAT)". *Atmospheric Chemistry and Physics*, Vol. 8, No. 22, pp. 6739–6753.  
   **DOI:** [10.5194/acp-8-6739-2008](https://doi.org/10.5194/acp-8-6739-2008)

4. **Horton, D. E., Skinner, C. B., Singh, D., & Diffenbaugh, N. S.** (2014). "Occurrence and persistence of future atmospheric stagnation events". *Nature Climate Change*, Vol. 4, pp. 698–703.  
   **DOI:** [10.1038/nclimate2272](https://doi.org/10.1038/nclimate2272)

5. **Wang, P., Cao, J., & Shen, Z.** (2018). "Spatial and seasonal variations of air stagnation and its impact on PM2.5 in China". *Atmospheric Environment*, Vol. 180, pp. 165–175.  
   **DOI:** [10.1016/j.atmosenv.2018.02.051](https://doi.org/10.1016/j.atmosenv.2018.02.051)

6. **Weiss, A. D.** (2001). "Topographic Position and Landforms Analysis". *ESRI International User Conference Proceedings*, San Diego, CA.

7. **Spencer, J. W.** (1971). "Fourier series representation of the position of the Sun". *Search*, Vol. 2, No. 5, p. 172.

8. **Reda, I., & Andreas, A.** (2004). "Solar position algorithm for solar radiation applications". *Solar Energy*, Vol. 76, No. 5, pp. 577–589.  
   **DOI:** [10.1016/j.solener.2003.12.003](https://doi.org/10.1016/j.solener.2003.12.003)

9. **Lawson, C. L., & Hanson, R. J.** (1974). *Solving Least Squares Problems*. Prentice-Hall, Englewood Cliffs, NJ, 340 pp. (Classics in Applied Mathematics, SIAM, 1995).

10. **Breiman, L.** (1996). "Stacked regressions". *Machine Learning*, Vol. 24, No. 1, pp. 49–64.  
    **DOI:** [10.1007/BF00117832](https://doi.org/10.1007/BF00117832)

11. **Wolpert, D. H.** (1992). "Stacked generalization". *Neural Networks*, Vol. 5, No. 2, pp. 241–259.  
    **DOI:** [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1)
