from marshmallow import fields

from lib.schema import BaseSchema, RoundedFloat


class HealthRiskQuerySchema(BaseSchema):
    user_id = fields.String(required=True)


class HealthRiskResponseSchema(BaseSchema):
    ckd_risk_score = RoundedFloat(
        places=2, description="CKD risk score", example=22.5, allow_none=True
    )
    cvd_risk_score = RoundedFloat(
        places=2, description="CVD risk score", example=25.5, allow_none=True
    )
    hypertension_risk_score = RoundedFloat(
        places=2, description="Hypertension risk score", example=25.5, allow_none=True
    )
    hyp_absolute_score = RoundedFloat(
        places=2,
        description="Absolute Hypertension score",
        example=25.5,
        allow_none=True,
    )
    diabetes_risk_score = RoundedFloat(
        places=2,
        description="Diabetes health risk score",
        example=25.5,
        allow_none=True,
    )
    overall_risk_score = RoundedFloat(
        places=2,
        description="Overall risk score based on CVD, Hypertension, and Diabetes risks",
        example=25.5,
        allow_none=True,
    )

    cvd_ckd_absolute_score = RoundedFloat(
        places=2,
        description="Absolute CVD score",
        example=25.5,
        allow_none=True,
    )
