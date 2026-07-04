-- Step 1: Identify CKD patients with comorbidities
WITH patient_diagnoses AS (
    SELECT subject_id,
           MAX(CASE WHEN icd_code LIKE 'N18%' OR icd_code LIKE '585%' THEN 1 ELSE 0 END) AS has_ckd,
           MAX(CASE WHEN icd_code LIKE 'E10%' OR icd_code LIKE 'E11%' OR icd_code LIKE '250%' THEN 1 ELSE 0 END) AS has_diabetes,
           MAX(CASE WHEN icd_code LIKE 'I10%' OR icd_code LIKE '401%' THEN 1 ELSE 0 END) AS has_hypertension,
           MAX(CASE WHEN icd_code LIKE 'I50%' OR icd_code LIKE '428%' THEN 1 ELSE 0 END) AS has_heart_failure
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    GROUP BY subject_id
),
ckd_patients AS (
    SELECT * FROM patient_diagnoses WHERE has_ckd = 1
),

-- Step 2: Extract longitudinal labs for the cohort
longitudinal_labs AS (
    SELECT l.subject_id, l.charttime,
           MAX(CASE WHEN l.itemid IN (50912, 52546) THEN l.valuenum END) AS creatinine_mg_dl,
           MAX(CASE WHEN l.itemid IN (50971, 52610) THEN l.valuenum END) AS potassium,
           MAX(CASE WHEN l.itemid IN (50983, 52623) THEN l.valuenum END) AS sodium,
           MAX(CASE WHEN l.itemid IN (51006) THEN l.valuenum END) AS bun,
           MAX(CASE WHEN l.itemid IN (50811, 51222) THEN l.valuenum END) AS hemoglobin,
           MAX(CASE WHEN l.itemid IN (50862) THEN l.valuenum END) AS albumin,
           MAX(CASE WHEN l.itemid IN (50970) THEN l.valuenum END) AS phosphorus,
           MAX(CASE WHEN l.itemid IN (50893) THEN l.valuenum END) AS calcium,
           MAX(CASE WHEN l.itemid IN (50882) THEN l.valuenum END) AS bicarbonate
    FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
    WHERE l.itemid IN (50912, 52546, 50971, 52610, 50983, 52623, 51006, 50811, 51222, 50862, 50970, 50893, 50882)
      AND l.subject_id IN (SELECT subject_id FROM ckd_patients)
    GROUP BY l.subject_id, l.charttime
),

-- Step 3: Filter for patients with sufficient data history (>=3 creatinine measurements over >=180 days)
eligible AS (
    SELECT subject_id, COUNT(creatinine_mg_dl) as n_labs,
           TIMESTAMP_DIFF(MAX(charttime), MIN(charttime), DAY) as span_days
    FROM longitudinal_labs
    WHERE creatinine_mg_dl IS NOT NULL AND creatinine_mg_dl BETWEEN 0.1 AND 30.0
    GROUP BY subject_id
    HAVING COUNT(creatinine_mg_dl) >= 3
       AND TIMESTAMP_DIFF(MAX(charttime), MIN(charttime), DAY) >= 180
)

SELECT p.subject_id, p.gender, p.anchor_age,
       c.has_diabetes, c.has_hypertension, c.has_heart_failure,
       l.charttime, l.creatinine_mg_dl, l.potassium, l.sodium, l.bun, l.hemoglobin, l.albumin, l.phosphorus, l.calcium, l.bicarbonate
FROM eligible e
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON e.subject_id = p.subject_id
JOIN ckd_patients c ON e.subject_id = c.subject_id
JOIN longitudinal_labs l ON e.subject_id = l.subject_id
WHERE l.creatinine_mg_dl IS NOT NULL
ORDER BY l.subject_id, l.charttime;
