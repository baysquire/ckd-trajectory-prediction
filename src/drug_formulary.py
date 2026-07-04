# the 20 focus drugs — one list for data, rl, and safety so names stay aligned.

DRUG_NAMES = [
    'lisinopril', 'metformin', 'empagliflozin', 'losartan', 'furosemide',
    'dapagliflozin', 'spironolactone', 'amlodipine', 'sodium bicarbonate',
    'allopurinol', 'ramipril', 'enalapril', 'irbesartan', 'valsartan',
    'hydrochlorothiazide', 'febuxostat', 'canagliflozin', 'finerenone',
    'atorvastatin', 'erythropoietin',
]

# mimic prescription strings can differ slightly from our canonical name
MIMIC_DRUG_ALIASES = {
    'lisinopril': 'lisinopril',
    'prinivil': 'lisinopril',
    'zestril': 'lisinopril',
    'metformin': 'metformin',
    'glucophage': 'metformin',
    'empagliflozin': 'empagliflozin',
    'jardiance': 'empagliflozin',
    'losartan': 'losartan',
    'cozaar': 'losartan',
    'furosemide': 'furosemide',
    'lasix': 'furosemide',
    'dapagliflozin': 'dapagliflozin',
    'farxiga': 'dapagliflozin',
    'spironolactone': 'spironolactone',
    'aldactone': 'spironolactone',
    'amlodipine': 'amlodipine',
    'norvasc': 'amlodipine',
    'sodium bicarbonate': 'sodium bicarbonate',
    'sod bicarb': 'sodium bicarbonate',
    'bicarbonate': 'sodium bicarbonate',
    'allopurinol': 'allopurinol',
    'zyloprim': 'allopurinol',
    'ramipril': 'ramipril',
    'altace': 'ramipril',
    'enalapril': 'enalapril',
    'vasotec': 'enalapril',
    'irbesartan': 'irbesartan',
    'avapro': 'irbesartan',
    'valsartan': 'valsartan',
    'diovan': 'valsartan',
    'hydrochlorothiazide': 'hydrochlorothiazide',
    'hctz': 'hydrochlorothiazide',
    'microzide': 'hydrochlorothiazide',
    'febuxostat': 'febuxostat',
    'uloric': 'febuxostat',
    'canagliflozin': 'canagliflozin',
    'invokana': 'canagliflozin',
    'finerenone': 'finerenone',
    'kerendia': 'finerenone',
    'atorvastatin': 'atorvastatin',
    'lipitor': 'atorvastatin',
    'erythropoietin': 'erythropoietin',
    'epo': 'erythropoietin',
    'epogen': 'erythropoietin',
    'procrit': 'erythropoietin',
}


def med_column(drug_name):
    """column name in the cohort csv for a drug flag."""
    return 'med_' + drug_name.replace(' ', '_')


def normalize_mimic_drug(raw_name):
    """map a mimic medication string to our canonical drug or None."""
    if not isinstance(raw_name, str):
        return None
    key = raw_name.strip().lower()
    if key in MIMIC_DRUG_ALIASES:
        return MIMIC_DRUG_ALIASES[key]
    for alias, canonical in MIMIC_DRUG_ALIASES.items():
        if alias in key:
            return canonical
    return None
