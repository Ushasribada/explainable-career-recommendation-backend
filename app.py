
import io
import re
from collections import Counter

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from docx import Document


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# DATA PATHS
# ============================================================

BASE_DIR = "data"

SKILLS_FILE = f"{BASE_DIR}/skills_list.csv"
JOB_ROLES_FILE = f"{BASE_DIR}/job_roles.csv"
JOB_DESCRIPTION_FILE = f"{BASE_DIR}/job_description.csv"


# ============================================================
# LOAD DATA
# ============================================================

skills_df = pd.read_csv(SKILLS_FILE)
job_roles_df = pd.read_csv(JOB_ROLES_FILE)
job_description = pd.read_csv(JOB_DESCRIPTION_FILE)

job_description = job_description.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)

skill_list = (
    skills_df["Skill Name"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+#./ -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(resume_text, skill_list):

    text = clean_text(resume_text)

    detected_skills = []

    for skill in skill_list:

        skill_clean = clean_text(skill)

        pattern = (
            r"(?<!\w)"
            + re.escape(skill_clean)
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            detected_skills.append(skill)

    return sorted(set(detected_skills))


# ============================================================
# JOB MARKET SKILL EXTRACTION
# ============================================================

job_description["Job Text"] = (
    job_description["Description"].fillna("").astype(str)
    + " "
    + job_description["Requirement"].fillna("").astype(str)
    + " "
    + job_description["Requirements"].fillna("").astype(str)
)

job_description["Extracted Skills"] = job_description[
    "Job Text"
].apply(
    lambda x: extract_skills(x, skill_list)
)


# ============================================================
# MARKET DEMAND
# ============================================================

skill_demand = Counter()

for skills in job_description["Extracted Skills"]:

    for skill in skills:

        skill_demand[skill] += 1


market_skills = pd.DataFrame(
    skill_demand.items(),
    columns=["Skill", "Job_Count"]
)

market_skills = market_skills.sort_values(
    by="Job_Count",
    ascending=False
).reset_index(drop=True)

market_skills["Demand_Percentage"] = (
    market_skills["Job_Count"]
    / len(job_description)
    * 100
).round(2)


market_demand_dict = dict(
    zip(
        market_skills["Skill"].str.lower(),
        market_skills["Demand_Percentage"]
    )
)


# ============================================================
# PDF / DOCX TEXT EXTRACTION
# ============================================================

def extract_resume_text(file_bytes, filename):

    filename = filename.lower()

    if filename.endswith(".pdf"):

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()


    elif filename.endswith(".docx"):

        document = Document(
            io.BytesIO(file_bytes)
        )

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        return text.strip()


    else:

        raise ValueError(
            "Unsupported file format. Please upload PDF or DOCX."
        )


# ============================================================
# REQUIRED SKILLS
# ============================================================

def parse_required_skills(required_skills):

    if pd.isna(required_skills):
        return []

    skills = str(required_skills).split("|")

    return [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]


# ============================================================
# SKILL GAP
# ============================================================

def calculate_skill_gap(
    candidate_skills,
    required_skills
):

    candidate_set = {
        str(skill).strip().lower()
        for skill in candidate_skills
    }

    required_set = {
        str(skill).strip().lower()
        for skill in required_skills
    }

    matching = candidate_set.intersection(
        required_set
    )

    missing = required_set - candidate_set

    if len(required_set) > 0:

        match_percentage = (
            len(matching)
            / len(required_set)
        ) * 100

    else:

        match_percentage = 0


    return (
        round(match_percentage, 2),
        sorted(matching),
        sorted(missing)
    )


# ============================================================
# ANALYZE ALL JOB ROLES
# ============================================================

def analyze_skill_gap(
    candidate_skills,
    job_roles
):

    results = []

    for _, row in job_roles.iterrows():

        required_skills = parse_required_skills(
            row["Required Skills"]
        )

        (
            match_percentage,
            matching,
            missing
        ) = calculate_skill_gap(
            candidate_skills,
            required_skills
        )

        results.append({

            "Job Role":
                row["Job Title"],

            "Category":
                row["Category"],

            "Skill Match (%)":
                match_percentage,

            "Matching Skills":
                matching,

            "Missing Skills":
                missing,

            "Required Skill Count":
                len(required_skills)

        })


    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="Skill Match (%)",
        ascending=False
    ).reset_index(drop=True)


    return result_df


# ============================================================
# MARKET SCORE
# ============================================================

def calculate_market_score(
    required_skills
):

    if not required_skills:
        return 0


    scores = []

    for skill in required_skills:

        skill_lower = (
            str(skill)
            .strip()
            .lower()
        )

        demand = market_demand_dict.get(
            skill_lower,
            0
        )

        scores.append(demand)


    return round(
        sum(scores) / len(scores),
        2
    )


# ============================================================
# DYNAMIC RECOMMENDATIONS
# ============================================================

def generate_dynamic_recommendations(
    candidate_skills,
    job_roles,
    skill_weight=0.70,
    market_weight=0.30
):

    skill_gap_df = analyze_skill_gap(
        candidate_skills,
        job_roles
    )

    recommendations = []


    for _, row in skill_gap_df.iterrows():

        matching_job = job_roles[
            job_roles["Job Title"]
            == row["Job Role"]
        ]

        if len(matching_job) == 0:
            continue


        required_skills = parse_required_skills(
            matching_job.iloc[0]["Required Skills"]
        )


        market_score = calculate_market_score(
            required_skills
        )

        skill_match = row["Skill Match (%)"]


        dynamic_score = (
            skill_weight * skill_match
            +
            market_weight * market_score
        )


        recommendations.append({

            "Job Role":
                row["Job Role"],

            "Category":
                row["Category"],

            "Skill Match (%)":
                skill_match,

            "Market Demand (%)":
                market_score,

            "Dynamic Score":
                round(dynamic_score, 2),

            "Matching Skills":
                row["Matching Skills"],

            "Missing Skills":
                row["Missing Skills"]

        })


    recommendation_df = pd.DataFrame(
        recommendations
    )


    recommendation_df = recommendation_df.sort_values(
        by="Dynamic Score",
        ascending=False
    ).reset_index(drop=True)


    return recommendation_df


# ============================================================
# EXPLAINABILITY
# ============================================================

def generate_explanation(row):

    matching_skills = row["Matching Skills"]
    missing_skills = row["Missing Skills"]

    match_percentage = row["Skill Match (%)"]
    market_percentage = row["Market Demand (%)"]
    dynamic_score = row["Dynamic Score"]

    matching_count = len(
        matching_skills
    )

    missing_count = len(
        missing_skills
    )


    if matching_count > 0:

        matching_text = ", ".join(
            skill.title()
            for skill in matching_skills
        )

    else:

        matching_text = (
            "No major matching skills"
        )


    if missing_count > 0:

        missing_text = ", ".join(
            skill.title()
            for skill in missing_skills
        )

    else:

        missing_text = (
            "No major missing skills"
        )


    explanation = (

        f"{row['Job Role']} is recommended because "

        f"{matching_count} required skills match "
        f"the candidate profile "
        f"({match_percentage:.2f}% skill match). "

        f"The average market demand of the "
        f"required skills is "
        f"{market_percentage:.2f}%. "

        f"Matching skills include: "
        f"{matching_text}. "

        f"To become better prepared for this role, "
        f"the candidate should develop: "
        f"{missing_text}. "

        f"The resulting dynamic recommendation "
        f"score is {dynamic_score:.2f}."

    )


    return explanation


# ============================================================
# LEARNING ROADMAP
# ============================================================

def normalize_skill(skill):

    return str(skill).strip().lower()


def calculate_skill_priority(
    missing_skills,
    market_demand_dict
):

    roadmap = []


    for skill in missing_skills:

        normalized_skill = normalize_skill(
            skill
        )

        market_demand = market_demand_dict.get(
            normalized_skill,
            0
        )


        roadmap.append({

            "Missing Skill":
                skill,

            "Market Demand (%)":
                market_demand

        })


    return pd.DataFrame(roadmap)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message":
            "Explainable AI Career Recommendation API is running.",

        "endpoint":
            "/analyze-file"

    })


# ============================================================
# MARKET TRENDS API
# ============================================================

@app.route("/market-trends", methods=["GET"])
def market_trends():

    try:

        trends = []

        for _, row in market_skills.head(15).iterrows():

            trends.append({
                "skill": str(row["Skill"]),
                "job_count": int(row["Job_Count"]),
                "demand_percentage": float(
                    row["Demand_Percentage"]
                )
            })

        return jsonify({
            "success": True,
            "trends": trends
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================
# RESUME ANALYSIS API
# ============================================================

@app.route(
    "/analyze-file",
    methods=["POST"]
)
def analyze_resume_file():

    try:

        if "resume" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No resume file uploaded."

            }), 400


        file = request.files["resume"]


        if file.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "No file selected."

            }), 400


        file_bytes = file.read()


        resume_text = extract_resume_text(
            file_bytes,
            file.filename
        )


        if (
            not resume_text
            or not resume_text.strip()
        ):

            return jsonify({

                "success": False,

                "error":
                    "Could not extract text from the resume."

            }), 400


        # ----------------------------------------
        # Extract candidate skills
        # ----------------------------------------

        candidate_skills = extract_skills(
            resume_text,
            skill_list
        )


        # ----------------------------------------
        # Generate recommendations
        # ----------------------------------------

        dynamic_results = (
            generate_dynamic_recommendations(
                candidate_skills,
                job_roles_df
            )
        )


        if (
            dynamic_results is None
            or len(dynamic_results) == 0
        ):

            return jsonify({

                "success": False,

                "error":
                    "No career recommendations could be generated."

            }), 500


        # ----------------------------------------
        # Explainable results
        # ----------------------------------------

        explainable_results = (
            dynamic_results.copy()
        )


        explainable_results[
            "Explanation"
        ] = explainable_results.apply(
            generate_explanation,
            axis=1
        )


        # ----------------------------------------
        # Top recommendation
        # ----------------------------------------

        top = explainable_results.iloc[0]


        matching_skills = (
            top["Matching Skills"]
        )

        missing_skills = (
            top["Missing Skills"]
        )


        # ----------------------------------------
        # Learning roadmap
        # ----------------------------------------

        roadmap_df = calculate_skill_priority(
            missing_skills,
            market_demand_dict
        )


        roadmap = []


        if (
            roadmap_df is not None
            and len(roadmap_df) > 0
        ):

            max_demand = (
                roadmap_df[
                    "Market Demand (%)"
                ].max()
            )


            if max_demand > 0:

                roadmap_df[
                    "Market Demand Score"
                ] = (

                    roadmap_df[
                        "Market Demand (%)"
                    ]

                    / max_demand

                ) * 100

            else:

                roadmap_df[
                    "Market Demand Score"
                ] = 0


            roadmap_df[
                "Gap Frequency Score"
            ] = 100


            roadmap_df[
                "Dynamic Priority"
            ] = (

                0.5
                * roadmap_df[
                    "Gap Frequency Score"
                ]

                +

                0.5
                * roadmap_df[
                    "Market Demand Score"
                ]

            )


            roadmap_df = roadmap_df.sort_values(
                by="Dynamic Priority",
                ascending=False
            )


            for _, row in roadmap_df.iterrows():

                roadmap.append({

                    "skill":
                        str(row["Missing Skill"]),

                    "market_demand":
                        float(
                            row["Market Demand (%)"]
                        ),

                    "priority":
                        float(
                            row["Dynamic Priority"]
                        )

                })


        # ----------------------------------------
        # Top 10 recommendations
        # ----------------------------------------

        recommendations = []


        for _, row in explainable_results.head(10).iterrows():

            recommendations.append({

                "job_role":
                    str(row["Job Role"]),

                "category":
                    str(row["Category"]),

                "skill_match":
                    float(row["Skill Match (%)"]),

                "market_demand":
                    float(row["Market Demand (%)"]),

                "dynamic_score":
                    float(row["Dynamic Score"]),

                "matching_skills":
                    row["Matching Skills"],

                "missing_skills":
                    row["Missing Skills"],

                "explanation":
                    str(row["Explanation"])

            })


        # ----------------------------------------
        # Final JSON response
        # ----------------------------------------

        response = {

            "success": True,

            "candidate": {

                "skills":
                    candidate_skills,

                "skill_count":
                    len(candidate_skills)

            },


            "recommendation": {

                "job_role":
                    str(top["Job Role"]),

                "skill_match":
                    float(
                        top["Skill Match (%)"]
                    ),

                "market_demand":
                    float(
                        top["Market Demand (%)"]
                    ),

                "dynamic_score":
                    float(
                        top["Dynamic Score"]
                    ),

                "matching_skills":
                    matching_skills,

                "missing_skills":
                    missing_skills,

                "explanation":
                    str(
                        top["Explanation"]
                    )

            },


            "learning_roadmap":
                roadmap,


            "top_recommendations":
                recommendations

        }


        return jsonify(response)


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
