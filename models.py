from pydantic import BaseModel

class LoanData(BaseModel):
    income: int
    credit_score: int
    loan_amount: int
    years_employed: int