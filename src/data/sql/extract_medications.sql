-- extract prescriptions for our focus drugs in the ckd cohort
-- maps to the 20 drugs in DRUG_NAMES from rl_env.py

WITH cohort AS (
    SELECT DISTINCT subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_code LIKE 'N18%' OR icd_code LIKE '585%'
)

SELECT p.subject_id, p.hadm_id,
       p.starttime, p.stoptime,
       p.medication, p.dose_val_rx, p.dose_unit_rx, p.route
FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
JOIN cohort c ON p.subject_id = c.subject_id
WHERE LOWER(p.medication) IN (
    -- ACE inhibitors
    'lisinopril', 'ramipril', 'enalapril',
    -- ARBs
    'losartan', 'irbesartan', 'valsartan',
    -- CCBs
    'amlodipine',
    -- diuretics
    'furosemide', 'hydrochlorothiazide', 'spironolactone',
    -- SGLT2 inhibitors
    'dapagliflozin', 'empagliflozin', 'canagliflozin',
    -- MRAs
    'finerenone',
    -- biguanides
    'metformin',
    -- urate lowering
    'allopurinol', 'febuxostat',
    -- statin
    'atorvastatin',
    -- supportive
    'sodium bicarbonate', 'erythropoietin', 'epoetin alfa'
)
ORDER BY p.subject_id, p.starttime;
