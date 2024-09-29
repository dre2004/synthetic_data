import logging
import sys

from models.aggregate import aggregate_health_risk
from models.cardiometabolic import estimate_cardiometabolic_risk
from models.hypertension import estimate_hypertension_risk
from lib.serializers import HealthRiskResponseSchema

logging.getLogger().addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)


def estimate_health_risk(hypertension_questions: dict, cvd_questions: dict, debug: bool = False):
    response = {
        "cvd_risk_score": -1,
        "hypertension_risk_score": -1,
        "diabetes_risk_score": -1,
        "overall_risk_score": -1,
    }

    hypertension_risk = None
    if hypertension_questions:
        try:
            abs_hyp_score, hypertension_risk = estimate_hypertension_risk(hypertension_questions)
            if hypertension_risk is not None:
                response["hypertension_risk_score"] = hypertension_risk
                response["hyp_absolute_score"] = abs_hyp_score
        except:
            if debug:
                logger.warning(
                    "Failed to estimate hypertension risk, will skip. {0}".format(
                        sys.exc_info()[0]
                    )
                )

    cardio_metabolic_risk = None
    if cvd_questions:
        try:
            abs_score, cardio_metabolic_risk = estimate_cardiometabolic_risk(cvd_questions)
            print("ABS SCORE: ", abs_score)
            if cardio_metabolic_risk is not None:
                response["ckd_risk_score"] = cardio_metabolic_risk["ckd"]
                response["cvd_risk_score"] = cardio_metabolic_risk["cvd"]
                response["diabetes_risk_score"] = cardio_metabolic_risk["diabetes"]
                response["cvd_ckd_absolute_score"] = abs_score
        except:
            if debug:
                logger.warning(
                    "Failed to estimate cardio metabolic risk, will skip. {0}".format(
                        sys.exc_info()[0]
                    )
                )

    if (
        hypertension_questions
        and cvd_questions
        and cardio_metabolic_risk is not None
        and hypertension_risk is not None
    ):
        response["overall_risk_score"] = aggregate_health_risk(
            cardio_metabolic_risk["composite"], hypertension_risk
        )

    serialized = HealthRiskResponseSchema().dump(response)

    return serialized
