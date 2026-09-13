# mongo.py
#
# MongoDB persistence layer for the Student Credit Risk app.
# This module ONLY stores/retrieves assessments — it never predicts risk.

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "student_credit_risk"
COLLECTION_NAME = "assessments"

_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    return _client


def get_collection():
    return get_client()[DB_NAME][COLLECTION_NAME]


def test_connection() -> bool:
    """Return True if MongoDB is reachable, False otherwise."""
    try:
        get_client().admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return False


def save_assessment(student_data: dict, assessment: dict, model_name: str) -> str:
    """Persist a student record + its Nemotron assessment. Returns the inserted id."""
    document = {
        "student_data": student_data,
        "assessment": assessment,
        "model": model_name,
        "created_at": datetime.now(timezone.utc),
    }
    result = get_collection().insert_one(document)
    return str(result.inserted_id)


def get_assessments(limit: int = 50) -> list:
    """Return the most recent assessments, newest first."""
    cursor = get_collection().find().sort("created_at", DESCENDING).limit(limit)
    return list(cursor)


def get_assessment_by_id(assessment_id: str) -> dict | None:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(assessment_id)
    except InvalidId:
        return None

    return get_collection().find_one({"_id": oid})


def get_analytics() -> dict:
    """Return simple aggregate stats over all stored assessments."""
    collection = get_collection()
    total = collection.count_documents({})

    if total == 0:
        return {
            "total": 0,
            "risk_level_counts": {},
            "loan_decision_counts": {},
            "average_risk_score": None,
        }

    risk_level_counts = {}
    for level in ("Low", "Moderate", "High"):
        risk_level_counts[level] = collection.count_documents({"assessment.risk_level": level})

    loan_decision_counts = {}
    for decision in ("Approve", "Investigate", "Disapprove"):
        loan_decision_counts[decision] = collection.count_documents({"assessment.loan_decision": decision})

    pipeline = [{"$group": {"_id": None, "avg_score": {"$avg": "$assessment.risk_score"}}}]
    agg_result = list(collection.aggregate(pipeline))
    average_risk_score = agg_result[0]["avg_score"] if agg_result else None

    return {
        "total": total,
        "risk_level_counts": risk_level_counts,
        "loan_decision_counts": loan_decision_counts,
        "average_risk_score": average_risk_score,
    }


if __name__ == "__main__":
    if test_connection():
        print(f"Connected to MongoDB at {MONGODB_URI}")
        print(f"Database: {DB_NAME}, Collection: {COLLECTION_NAME}")
        print(f"Stored assessments: {get_collection().count_documents({})}")
    else:
        print(f"Could NOT connect to MongoDB at {MONGODB_URI}")
        print("Make sure MongoDB is installed and running locally.")
