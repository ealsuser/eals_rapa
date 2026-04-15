import pandas as pd
from traitlets import List
import config
import datetime
import os
import json
import numpy as np

def load_alsfrsr_data(clean=True):

    df_alsfrsr = pd.read_csv(config.Paths.alsfrsr, sep=';', parse_dates=["created_at"])
    df_alsfrsr["date_created_at"] = df_alsfrsr.created_at.dt.date

    # Remove mislabeled rows
    if clean:
        df_alsfrsr = df_alsfrsr.query('ALS_total > 0').copy()

    # Creation of bulbar_sub_score
    df_alsfrsr["bulbar_subscore"] = (
        df_alsfrsr["speech"] + df_alsfrsr["salvation"] + df_alsfrsr["swallowing"]
    )

    # Creation of respiratory_sub_score
    df_alsfrsr["respiratory_subscore"] = (
        df_alsfrsr["dyspnea"]
        + df_alsfrsr["orthopnea"]
        + df_alsfrsr["respiratory_insufficiency"]
    )

    # Creation of fine_motor_sub_score and gross_motor_sub_score
    df_alsfrsr['fine_motor_subscore'] = df_alsfrsr['cutting_food_a'] + df_alsfrsr['cutting_food_b'] + df_alsfrsr['dressing_and_hygiene'] + df_alsfrsr['handwriting']
    df_alsfrsr['gross_motor_subscore'] = df_alsfrsr['turning_in_bed'] + df_alsfrsr['walking'] + df_alsfrsr['climbing_stairs']

    df_alsfrsr["session_id"] = df_alsfrsr.apply(
        lambda row: f"{row['eals_id']}__{row['date_created_at']}", axis=1
    )

    # Format
    df_alsfrsr.rename(
        {"date_created_at": "date", "eals_id": "user_id"}, axis=1, inplace=True
    )

    # Compute months since the first session per user (vectorized)
    first_date = df_alsfrsr.groupby('user_id')['date'].transform('min')
    months = (df_alsfrsr['date'] - first_date) / np.timedelta64(30, 'D')  # ~months as float
    df_alsfrsr.loc[:, 'months_since_first_session'] = months
    df_alsfrsr.loc[:, 'years_since_first_session'] = df_alsfrsr['months_since_first_session'] / 12.0

    return df_alsfrsr

DEMOGRAPHICS_COLUMN_MAP = {
    "eals_id": "user_id",
    ", please indicate the relationship (mother, father, sibling etc…), the disease, and the approximate age of onset of disease symptoms": "family_disease_details",
    "ADHD/ADD": "adhd_add",
    "ALS": "als",
    "Anxiety": "anxiety",
    "Bipolar disorder": "bipolar_disorder",
    "Brain Aneurysm": "brain_aneurysm",
    "C9ORF mutation": "c9orf_mutation",
    "Cancer": "cancer",
    "City, State and Country of Residence": "residence_location",
    "City, State and Country of birth": "birth_location",
    "Clinic or Hospital city": "clinic_city",
    "Clinic or Hospital name": "clinic_name",
    "Cognitive Impairment": "cognitive_impairment",
    "Concussion (s)": "concussions",
    "Consent form signed?": "consent_signed",
    "Date Of Birth": "date_of_birth",
    "Date Of Diagnosis": "date_of_diagnosis",
    "Date of Symptom Onset": "symptom_onset_date",
    "Dementia": "dementia",
    "Depression": "depression",
    "Diabetes": "diabetes",
    "Diagnosis": "diagnosis",
    "Do any of the following apply to you?": "additional_conditions",
    "Do you have a cold right now?": "current_cold",
    "Do you have a positive family history for ALS or any other neurodegenerative disease (Frontotemporal Dementia, Parkinson’s Disease, Altzheimer’s Disease)?": "family_neuro_history",
    "Do you have stairs at home?": "stairs_at_home",
    "Epilepsy": "epilepsy",
    "Ethnicity": "ethnicity",
    "Have you been diagnosed with hearing loss?": "hearing_loss",
    "Have you been hospitalized in the past 3 months?": "recent_hospitalization",
    "Have you ever received therapy to improve your speech?": "speech_therapy",
    "Have you quit smoking in the past year?": "recent_quit_smoking",
    "HeightFeet": "height_feet",
    "HeightInch": "height_inch",
    "Highest level of Education": "education_level",
    "How are your stairs at your house?": "stairs_description",
    "How many steps are there?": "number_of_steps",
    "Is English your first language?": "english_first_language",
    "Limb": "symptom_onset_site_limb",
    "Marital status": "marital_status",
    "Muscle Disease": "muscle_disease",
    "Physical gender at birth": "birth_gender",
    "Please List Specific Allergies": "specific_allergies",
    "Please measure the height of a step": "step_height",
    "Please specify the dimensions": "specified_dimensions",
    "Race": "race",
    "Residence Type": "residence_type",
    "Residence type": "residence_type_2",  # Duplicate, consider if needed
    "Schizophrenia": "schizophrenia",
    "Site of Symptom Onset": "symptom_onset_site",
    "Smoking History": "smoking_history",
    "Total Household Income": "household_income",
    "Weight": "weight",
    "What are your current prescriptions": "current_prescriptions",
    "What is the flooring of your stairs?": "stairs_flooring",
    "What is your dominant hand?": "dominant_hand",
    "What is your occupation?": "occupation",
    "What type of Orthotics do you use?": "orthotics_type",
    "What type of shoes will you be wearing for the entire duration of the study? ": "study_shoes_type",
    "Which region in the United States do you consider yourself from?": "us_region_identity",
    "Your Neurologist or Physician name": "neurologist_physician_name",
    "other education": "other_education",
    "other race": "other_race",
    "other symptoms": "other_symptoms",
}

DEMOGRAPHICS_FEATURES = ["bmi", "age", "years_since_onset"]

def add_features_to_demographics(
    df_demographics: pd.DataFrame, features: List[str] = DEMOGRAPHICS_FEATURES
) -> pd.DataFrame:
    if "bmi" in features:
        df_demographics["height_feet"] = pd.to_numeric(
            df_demographics["height_feet"], errors="coerce"
        )
        df_demographics["height_inch"] = pd.to_numeric(
            df_demographics["height_inch"], errors="coerce"
        )
        df_demographics["weight"] = pd.to_numeric(
            df_demographics["weight"], errors="coerce"
        )

        df_demographics["height_in_m"] = (
            df_demographics["height_feet"] * 12 + df_demographics["height_inch"]
        ) * 0.0254
        df_demographics["weight_in_kg"] = df_demographics["weight"] * 0.453592
        df_demographics["bmi"] = df_demographics["weight_in_kg"] / (
            df_demographics["height_in_m"] ** 2
        )

    if "age" in features:
        df_demographics["date_of_birth"] = pd.to_datetime(
            df_demographics["date_of_birth"]
        )
        current_year = pd.to_datetime(df_demographics['date']).dt.year
        df_demographics["age"] = (
            current_year - df_demographics["date_of_birth"].dt.year
        )

    if "years_since_onset" in features:
        df_demographics["symptom_onset_date"] = pd.to_datetime(
            df_demographics["symptom_onset_date"]
        )
        df_demographics["years_since_onset"] = (
            pd.to_datetime(df_demographics['date']) - df_demographics["symptom_onset_date"]
        ).dt.days / 365.25

    return df_demographics

def load_demographics_data():

     # Read CSV file
    df_demographics = pd.read_csv(
        config.Paths.demographics, sep=";", parse_dates=["date_created", "date_updated"]
    )
    df_demographics.rename(columns={'eals_id':'user_id'}, inplace=True)
    
    # Convert 'date_created' to datetime if it exists
    df_demographics["date"] = pd.to_datetime(df_demographics["date_created"]).dt.date
    df_demographics_date = df_demographics[['user_id', 'date']].drop_duplicates()

    # Drop unnecessary columns
    drop_columns = [
        "parent_field",
        "date_created",
        "date_updated",
        "sub_id",
        "demographics_id",
    ]
    df_demographics.drop(
        [col for col in drop_columns if col in df_demographics.columns],
        axis=1,
        inplace=True,
    )
    
    # Pivot the DataFrame
    df_demographics = df_demographics.pivot_table(
        index="user_id", columns="field", values="value", aggfunc="first"
    )
    df_demographics.reset_index(inplace=True)
    
    # Rename columns
    df_demographics.rename(columns=DEMOGRAPHICS_COLUMN_MAP, inplace=True)

    df_demographics = df_demographics.merge(
        df_demographics_date, on="user_id", how="left"
    )
    df_demographics = add_features_to_demographics(df_demographics)
    df_demographics["session_id"] = df_demographics.apply(
        lambda row: f"{row['user_id']}__{row['date']}", axis=1
    )
    df_demographics = df_demographics.groupby('user_id').first().reset_index()
    return df_demographics

def load_roads_data():
    df_roads = pd.read_csv(config.Paths.roads, sep=';')
    df_roads.rename(columns={'eals_id':'user_id'}, inplace=True)
    df_roads["date"] = pd.to_datetime(df_roads.date_created).dt.date
    df_roads['session_id'] = df_roads['user_id'].astype(str) + '__' + df_roads['date'].astype(str)
    return df_roads

def print_data_info(prefix, df):
    print(
        f"{prefix:<50} subjects: {df.user_id.nunique():>6}, sessions: {df.session_id.nunique():>6}, spiro: {df.seriesID.nunique():>6}, cols: {df.shape[1]:>3}"
    )

def load_zephyrx_data(
    at_least_efforts_fvc_usable=2,
    keep_usable_sessions_only=True,
    at_least_months_in_study=2,
    at_least_n_sessions=6,
    multiply_by_100=False,
):
    """
    Load and preprocess ZephyrX data from JSON files.

    This function loads data from the specified path, processes it to extract
    relevant columns and features, applies various filters for data quality,
    and returns a cleaned DataFrame for further analysis.

    Args:
        preprocess_data (bool, optional): Whether to apply preprocessing steps,
            such as excluding test users and invalid sessions. Default is True.
        only_at_rest (bool, optional): Whether to retain only "at rest" spirometries.
            Default is True.
        at_least_efforts_fvc_usable (int, optional): Minimum number of usable FVC efforts
            required to include a session. Default is 2.
        keep_usable_sessions_only (bool, optional): Whether to retain only sessions
            that meet usability criteria (`n_fvc_atleast_usable` or `vc > 0`).
            Default is True.
        at_least_months_in_study (int, optional): Minimum number of months a subject
            must have been in the study to be included. Default is 2.
        at_least_n_sessions (int, optional): Minimum number of sessions required
            per subject. Default is 6.
        drop_pulmonologist (bool, optional): Whether to exclude sessions before
            the pulmonologist intervention on "2023-03-01". Default is True.
        multiply_by_100 (bool, optional): Whether to scale `fvcPercPred_2019`
            and `vcPercPred` variables by 100. Default is False.
        replace_0 (bool, optional): Whether to replace zero values in critical
            variables (`vc`, `vcPercPred`, `fvc_2019`, `fvcPercPred_2019`)
            with NaN. Default is False. Se mantiene solo por compatibilidad con el anterior.
            #VALIDATE con Feli

    Returns:
        pd.DataFrame: A cleaned and preprocessed DataFrame containing ZephyrX data.

    Notes:
        - IMPORTANT: To keep almost all sessions (including those that do not meet usability
          criteria), set `keep_usable_sessions_only` to False and set the args
          `at_least_months_in_study` and `at_least_n_sessions` to 0. Warning, the regression columns
          will be computed for all subject with at least 3 sessions, the rest will have NaNs in the
          regression columns.
        - Time reference: There are two levels of 0 reference for time
            - first session: 0 is the date of the first session of enrollment (days_since_first_session)
            - first valid session: 0 is the first date that survives the quality cleaning (days_since_first_valid_session)
        - The function processes and filters data incrementally, reporting the
          dimensions of the DataFrame at each step.
        - Regression columns for specific spirometry targets (`fvcPercPred_2019`
          and `vcPercPred`) can be computed based on sufficient sessions (hardcoded to 3).
        - Data usability is determined by criteria applied to `n_fvc_atleast_usable`
          and `vc > 0` in.
        - patientID is the indicator from zephyrx meanwhile user_id is our internal id across tables

    Examples:
        >>> df_cleaned = load_zephyrx_data_paper(
                path="path/to/zephyrx/data",
                preprocess_data=True,
                at_least_efforts_fvc_usable=3,
                at_least_n_sessions=5,
                drop_pulmonologist=False
            )
        >>> print(df_cleaned.head())
    """
    # Load data
    files_tests = os.listdir(config.Zephyrx.raw)

    # Load tests
    tests = []
    for file in files_tests:
        with open(config.Zephyrx.raw + file) as f:
            data = json.load(f)
        tests.append(data)
    df_zephyrx = pd.DataFrame(tests)

    # New columns
    df_zephyrx.rename({"studyID": "user_id"}, axis=1, inplace=True)

    # Parse time cols with to_datetime
    df_zephyrx["created"] = pd.to_datetime(df_zephyrx["created"], utc=True)
    df_zephyrx = df_zephyrx.sort_values(["user_id", "created"])

    # Proctoring
    df_zephyrx["is_proctored"] = df_zephyrx["coachedSessionID"] != ""

    # Column "date" (excluding hours)
    df_zephyrx["date"] = pd.to_datetime(df_zephyrx["created"].dt.date)

    # Days since first session
    df_zephyrx["days_since_first_session"] = df_zephyrx.groupby("patientID")[
        "date"
    ].transform(lambda x: (x - x.min()).dt.days)

    # Compute months since the first session per user (vectorized)
    first_date = df_zephyrx.groupby('user_id')['date'].transform('min')
    months = (df_zephyrx['date'] - first_date) / np.timedelta64(30, 'D')  # ~months as float
    df_zephyrx.loc[:, 'months_since_first_session'] = months

    # Create a unique Session ID column
    df_zephyrx["session_id"] = df_zephyrx.apply(
        lambda row: f"{row['user_id']}__{row['date'].date()}", axis=1
    )

    # Add information on quality of session's efforts (depends only on the session)
    df_zephyrx["n_fvc_atleast_usable"] = df_zephyrx.efforts.apply(
        lambda efs: len(
            [
                e
                for e in efs
                if "fvcAcceptability_2019" in e.keys()
                and e["fvcAcceptability_2019"] != "NotAcceptable"
            ]
        )
    )

    # Multiply by 100 PercPred vars if needed
    if multiply_by_100:
        df_zephyrx["fvcPercPred_2019"] *= 100
        df_zephyrx["vcPercPred"] *= 100
        print("--- SCALING PERC PRED VARS TO PERCENTAGE: fvcPercPred_2019, vcPercPred")

    # Numerate dates for each user
    df_zephyrx["date_numeration"] = (
        df_zephyrx.groupby("user_id")["date"].rank(method="dense").astype(int)
    )

    # Numerate turns for each user's session
    df_zephyrx["turn_numeration"] = (
        df_zephyrx.groupby(["user_id", "date_numeration"]).cumcount() + 1
    )

    # Process data
    step = 1
    print_data_info(f"{step} - ORIGINAL DIMENSIONS:", df_zephyrx)

    # SESSION VALIDITY: Create dummy column for usability
    # Same criteria verbosed:
    crit = f"n_fvc_atleast_usable >= {at_least_efforts_fvc_usable} or vc > 0"
    df_zephyrx["eals_usability"] = df_zephyrx.eval(crit).astype(int)
    if keep_usable_sessions_only:
        step += 1
        df_zephyrx = df_zephyrx.query("eals_usability == 1")
        print_data_info(f"{step} - CRITERIA: {crit}", df_zephyrx)
    else:
        print(
            "--- KEEP SESSIONS THAT MAY NOT MET USABILITY CRITERIA (eals_usability column)"
        )

    # Compute timespan for each user, dependes on wheter we keep or not the sessions that not met usability criteria
    print(
        f"--- WARNING computing 'total_timespan' for each user after usability criteria: {keep_usable_sessions_only}"
    )
    df_zephyrx = df_zephyrx.copy()
    df_zephyrx["total_timespan"] = df_zephyrx.groupby("user_id")[
        "date"
    ].transform(lambda x: (x.max() - x.min()).days).copy()
    print(
        f"--- TOTAL TIMESPAN: min {df_zephyrx.total_timespan.min()}, max {df_zephyrx.total_timespan.max()}, mean {df_zephyrx.total_timespan.mean():.2f}"
    )

    # Filter subjects at_least_months_in_study, dependes on wheter we keep or not the sessions that not met usability criteria
    if at_least_months_in_study > 0:
        step += 1
        df_zephyrx = df_zephyrx.query(
            "total_timespan >= (@at_least_months_in_study * 30)"
        )
        print_data_info(
            f"{step} - MORE THAN {at_least_months_in_study} MONTHS IN STUDY", df_zephyrx
        )
    else:
        print("--- NOT FILTERING SUBJECTS BY MINIMUM MONTHS IN STUDY")
    
    df_zephyrx = df_zephyrx.copy()
    df_zephyrx["at_least_n_usable"] = df_zephyrx.groupby(["user_id", "pftType"])[
        "eals_usability"
    ].transform("sum")
    if at_least_n_sessions > 0:
        step += 1
        df_zephyrx = df_zephyrx.query("at_least_n_usable >= @at_least_n_sessions")
        print_data_info(
            f"{step} - SUBJECTS MORE THAN {at_least_n_sessions} SESSIONS pftType",
            df_zephyrx,
        )
    else:
        print("--- NOT FILTERING SUBJECTS BY MINIMUM NUMBER OF SESSIONS")

    return df_zephyrx

################################################################################################################ AURAL
def extract_date(input_string):
    first_chunk = input_string.split("-")[0]
    try:
        date = datetime.datetime.strptime(first_chunk, "%Y%m%d")
        return date
    except ValueError:
        return None
def get_json_steps_meta_file(file_list):
    json_list = [file for file in file_list if file.endswith("_steps_meta.json")]
    if len(json_list) > 0:
        return json_list[0]
    else:
        return None
contains_json = lambda file_list: any(file.endswith(".json") for file in file_list)
contains_wav = lambda file_list: any(file.endswith(".wav") for file in file_list)

def load_aural_data(regenerate=False):
    """
    Load Aural Analytics.
    """

    if not regenerate:
        return pd.read_csv(config.Aural.preprocessed + "df_aural.csv")

    # Each folder corresponds to a user_id
    aural_user_ids = [
        x for x in os.listdir(config.Aural.raw) if not (x.startswith(".") or x.startswith("eals"))
    ]

    # Obtain a df with each folder
    res = []
    for user_id in aural_user_ids:
        all_folders = list(
            filter(lambda x: not x.startswith("."), os.listdir(config.Aural.raw + "/" + user_id))
        )
        for folder in all_folders:
            all_files = os.listdir(config.Aural.raw + "/" + user_id + "/" + folder)
            res.append([user_id, folder, all_files, config.Aural.raw + "/" + user_id + "/" + folder])
    df_aural_folders = pd.DataFrame(res, columns=["user_id", "folder", "files", "path"])

    # Add metadata and flags
    df_aural_folders["starts_with_date"] = df_aural_folders.folder.apply(
        lambda x: x.startswith("202")
    )
    df_aural_folders["date"] = df_aural_folders.folder.apply(extract_date)
    df_aural_folders["contains_json"] = df_aural_folders.files.apply(contains_json)
    df_aural_folders["contains_wav"] = df_aural_folders.files.apply(contains_wav)
    df_aural_folders["json_steps_meta_file"] = df_aural_folders.files.apply(
        get_json_steps_meta_file
    )
    df_aural_folders["number_of_files"] = df_aural_folders.files.apply(lambda x: len(x))

    # Drop those with incomplete data
    df_aural_folders = df_aural_folders.query("contains_json and starts_with_date and contains_wav").copy()
    df_aural_folders.dropna(inplace=True)

    # Obtein each session's turns
    df_json_steps_meta = []
    df_aural_folders["error"] = False
    for i, row in df_aural_folders.iterrows():
        try:
            with open(row.path + "/" + row.json_steps_meta_file) as f:
                data_json_meta = json.load(f)
                df_json_steps_meta.append(pd.DataFrame(data_json_meta))
        except:
            print(f"Error with {i} row (session)")
            df_aural_folders.loc[i, "error"] = True
    df_json_steps_meta = pd.concat(df_json_steps_meta, ignore_index=True)

    # Rename
    df_aural_folders.rename(
        columns={"user_id": "participantId", "folder": "sessionId"}, inplace=True
    )
    print(f"Orginal aural dimensions: {df_aural_folders.shape[0]}")

    # Add wav name and its path
    df_aural = df_json_steps_meta.copy()
    df_aural["wav"] = None
    df_aural["path"] = None
    for i, row in df_aural.iterrows():
        df_folder_session = df_aural_folders.query(
            "participantId == @row.participantId and sessionId == @row.sessionId"
        )
        path = df_folder_session.iloc[0].path
        files_list = df_folder_session.iloc[0].files
        if row["stepId"] + ".wav" in files_list:
            df_aural.loc[i, "path"] = path + "/" + row["stepId"] + ".wav"
            df_aural.loc[i, "wav"] = row["stepId"] + ".wav"

    # Add relative wav path
    df_aural["relative_wav_path"] = df_aural.path.apply(
        lambda x: x.replace(config.Aural.raw + "/", "") if isinstance(x, str) else None
    )

    # Column "date_only" (excluding hours)
    df_aural["startTime"] = pd.to_datetime(df_aural["startTime"], utc=True)
    df_aural["date_only"] = df_aural["startTime"].dt.date

    # Drop users with no Wav path
    df_aural = df_aural.dropna(subset=["path"])
    df_aural.reset_index(drop=True, inplace=True)
    print(f"Drop rows with no wav path: {df_aural.shape[0]}")

    # Format
    df_aural = df_aural.copy()
    df_aural["session_id"] = df_aural.apply(
        lambda row: f"{row['participantId']}__{row['date_only']}", axis=1
    )
    df_aural.rename(
        columns={"participantId": "user_id", "date_only": "date"}, inplace=True
    )

    if not os.path.exists(config.Aural.preprocessed):
        os.makedirs(config.Aural.preprocessed)
    df_aural.to_csv(config.Aural.preprocessed + "df_aural.csv", index=False)

    return df_aural