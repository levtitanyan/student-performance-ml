import pandas as pd
from sklearn.preprocessing import StandardScaler


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to more descriptive, human-readable names.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with renamed columns.
    """
    rename_map = {
        "famsize": "family_size",
        "Pstatus": "parent_status",
        "Medu": "mother_education",
        "Fedu": "father_education",
        "Mjob": "mother_job",
        "Fjob": "father_job",
        "reason": "school_choice_reason",
        "traveltime": "travel_time",
        "studytime": "study_time",
        "failures": "past_failures",
        "schoolsup": "school_support",
        "famsup": "family_support",
        "paid": "extra_paid_classes",
        "nursery": "attended_nursery",
        "higher": "wants_higher_edu",
        "internet": "internet_access",
        "romantic": "in_romantic_relationship",
        "famrel": "family_relationship_quality",
        "freetime": "free_time",
        "goout": "social_life",
        "Dalc": "weekday_alcohol_use",
        "Walc": "weekend_alcohol_use",
        "health": "health_status",
        "G1": "grade_T1",
        "G2": "grade_T2",
        "G3": "final_grade"
    }

    return df.rename(columns=rename_map)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using a hybrid strategy:
    - Binary categorical variables are mapped to 0/1
    - Multiclass features with known 'other' values are one-hot encoded for selected key categories only.
      'Other' is implicitly represented when all corresponding binary columns are 0.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """

    # Binary replacement mapping
    binary_map = {
        'yes': 1, 'no': 0,
        'F': 0, 'M': 1,
        'T': 1, 'A': 0,
        'U': 1, 'R': 0,
        'GT3': 1, 'LE3': 0,
        'GP': 1, 'MS': 0  
    }

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Handle binary categorical columns
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    df[binary_cols] = df[binary_cols].replace(binary_map)

    # Handle multiclass columns manually (compact one-hot encoding)
    # Guardian → mother/father (other is implicit)
    df["guardian_mother"] = (df["guardian"] == "mother").astype(int)
    df["guardian_father"] = (df["guardian"] == "father").astype(int)
    df.drop(columns=["guardian"], inplace=True)

    # mother_job → teacher, health, services
    df["mother_job_teacher"] = (df["mother_job"] == "teacher").astype(int)
    df["mother_job_health"] = (df["mother_job"] == "health").astype(int)
    df["mother_job_services"] = (df["mother_job"] == "services").astype(int)
    df.drop(columns=["mother_job"], inplace=True)

    # father_job → teacher, services
    df["father_job_teacher"] = (df["father_job"] == "teacher").astype(int)
    df["father_job_services"] = (df["father_job"] == "services").astype(int)
    df.drop(columns=["father_job"], inplace=True)

    # school_choice_reason → course, reputation, home
    df["school_choice_reason_course"] = (df["school_choice_reason"] == "course").astype(int)
    df["school_choice_reason_reputation"] = (df["school_choice_reason"] == "reputation").astype(int)
    df["school_choice_reason_home"] = (df["school_choice_reason"] == "home").astype(int)
    df.drop(columns=["school_choice_reason"], inplace=True)

    return df




def scale_features(df: pd.DataFrame, exclude: list = ["student_id", "final_grade"]) -> pd.DataFrame:
    """
    Standardize numerical features using z-score scaling (mean = 0, std = 1).
    Excludes identifier and target columns by default.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical and categorical features.
        exclude (list): Columns to exclude from scaling (e.g., ID and target).

    Returns:
        pd.DataFrame: DataFrame with scaled numerical features.
    """
    # Select numeric columns, excluding specified ones
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(exclude)

    # Initialize and apply scaler
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"Scaled {len(num_cols)} numerical features.")
    return df
