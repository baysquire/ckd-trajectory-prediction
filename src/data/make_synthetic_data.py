import pandas as pd
import numpy as np
import os

def create_fake_data(num_patients=200):
    # generate fake cohort
    np.random.seed(42)

    rows = []

    for pid in range(num_patients):
        age = np.random.randint(40, 85)
        is_female = np.random.choice([0, 1])
        gender = 'F' if is_female else 'M'

        # comorbidities
        has_diabetes = int(np.random.random() < (0.3 + age * 0.003))
        has_hypertension = int(np.random.random() < (0.4 + age * 0.004))
        has_heart_failure = int(np.random.random() < (0.1 + age * 0.002))

        # starting labs
        base_cr = np.random.uniform(0.8, 4.0)
        base_potassium = np.random.uniform(3.5, 5.5)
        base_sodium = np.random.uniform(135, 145)
        base_bun = base_cr * np.random.uniform(8, 15)
        base_hemoglobin = np.random.uniform(9, 16) - (base_cr * 0.5)
        base_albumin = np.random.uniform(2.5, 4.5)
        base_phosphorus = np.random.uniform(2.5, 5.5)
        base_calcium = np.random.uniform(8.0, 10.5)
        base_bicarbonate = np.random.uniform(18, 28)

        num_visits = np.random.randint(4, 12)

        for visit in range(num_visits):
            # simulate labs
            cr = max(0.5, base_cr + np.random.normal(0, 0.15))
            potassium = max(2.5, min(7.0, base_potassium + np.random.normal(0, 0.2)))
            sodium = max(125, min(155, base_sodium + np.random.normal(0, 1.5)))
            bun = max(5, base_bun + np.random.normal(0, 2))
            hgb = max(5, min(18, base_hemoglobin + np.random.normal(0, 0.4)))
            albumin = max(1.5, min(5.5, base_albumin + np.random.normal(0, 0.1)))
            phos = max(1.5, min(8, base_phosphorus + np.random.normal(0, 0.2)))
            calcium = max(6.0, min(12, base_calcium + np.random.normal(0, 0.2)))
            bicarb = max(10, min(35, base_bicarbonate + np.random.normal(0, 1)))

            # simulate meds
            on_lisinopril = int(has_hypertension and np.random.random() < 0.4 and cr < 3)
            on_metformin = int(has_diabetes and np.random.random() < 0.5 and cr < 2)
            on_empagliflozin = int(np.random.random() < 0.3 and cr < 3)
            on_losartan = int(has_hypertension and np.random.random() < 0.3 and not on_lisinopril)
            on_furosemide = int(has_heart_failure and np.random.random() < 0.5)
            on_dapagliflozin = int(np.random.random() < 0.15 and not on_empagliflozin and cr < 3)
            on_spironolactone = int(has_heart_failure and np.random.random() < 0.2 and potassium < 5.0)
            on_amlodipine = int(has_hypertension and np.random.random() < 0.3)
            on_sodium_bicarbonate = int(bicarb < 22 and np.random.random() < 0.3)
            on_allopurinol = int(np.random.random() < 0.15)

            rows.append({
                'subject_id': pid,
                'time_idx': visit,
                'anchor_age': age,
                'gender': gender,
                'diabetes': has_diabetes,
                'hypertension': has_hypertension,
                'heart_failure': has_heart_failure,
                'creatinine_mg_dl': round(cr, 2),
                'potassium': round(potassium, 2),
                'sodium': round(sodium, 1),
                'bun': round(bun, 1),
                'hemoglobin': round(hgb, 1),
                'albumin': round(albumin, 2),
                'phosphorus': round(phos, 2),
                'calcium': round(calcium, 2),
                'bicarbonate': round(bicarb, 1),
                'med_lisinopril': on_lisinopril,
                'med_metformin': on_metformin,
                'med_empagliflozin': on_empagliflozin,
                'med_losartan': on_losartan,
                'med_furosemide': on_furosemide,
                'med_dapagliflozin': on_dapagliflozin,
                'med_spironolactone': on_spironolactone,
                'med_amlodipine': on_amlodipine,
                'med_sodium_bicarbonate': on_sodium_bicarbonate,
                'med_allopurinol': on_allopurinol,
            })

            # gradual progression
            base_cr += np.random.uniform(0.02, 0.08)
            # potassium drift
            base_potassium += 0.01
            # hemoglobin drift
            base_hemoglobin -= 0.02

    df = pd.DataFrame(rows)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    out_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'synthetic_cohort.csv')
    df.to_csv(out_path, index=False)
    print(f"saved {len(df)} records for {df['subject_id'].nunique()} patients to {out_path}")

if __name__ == '__main__':
    create_fake_data()
