"""
data_generator.py
------------------
Generates a realistic, telecom-style "Customer Churn" dataset.

Why generate instead of download?
This environment has no internet access to public dataset repositories,
so instead of a broken download link, we synthesize data with the same
structure, feature relationships, and "messiness" (missing values, mixed
types) as the well-known Telco Customer Churn dataset. Churn probability
is built from realistic business logic (contract type, tenure, charges,
support calls) rather than random noise, so the ML models trained later
actually learn meaningful patterns -- just like on real data.

Run:
    python data_generator.py
"""

import numpy as np
import pandas as pd

from utils import RAW_DATA_PATH, ensure_dirs, print_section

RANDOM_SEED = 42
N_CUSTOMERS = 5000


def generate_dataset(n=N_CUSTOMERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{100000 + i}" for i in range(n)]
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

    tenure_months = rng.integers(0, 73, size=n)  # 0 to 72 months
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.29, 0.71])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
    )

    base_charge = rng.normal(65, 20, size=n).clip(18, 120)
    monthly_charges = np.round(base_charge, 2)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.9, 1.05, n), 2)

    num_support_calls = rng.poisson(1.5, size=n)

    # ------------------------------------------------------------------
    # Build churn probability from realistic business logic
    # ------------------------------------------------------------------
    churn_score = np.zeros(n)

    # Month-to-month customers churn far more than long contracts
    churn_score += np.where(contract_type == "Month-to-month", 0.35, 0)
    churn_score += np.where(contract_type == "One year", 0.08, 0)

    # New customers (low tenure) churn more
    churn_score += np.where(tenure_months < 6, 0.25, 0)
    churn_score += np.where((tenure_months >= 6) & (tenure_months < 18), 0.10, 0)
    churn_score -= np.where(tenure_months > 48, 0.15, 0)

    # Fiber optic customers churn more (often due to price complaints)
    churn_score += np.where(internet_service == "Fiber optic", 0.15, 0)

    # No tech support increases churn
    churn_score += np.where(tech_support == "No", 0.12, 0)

    # High monthly charges slightly increase churn
    churn_score += (monthly_charges - 65) / 300

    # Many support calls strongly increase churn (frustration signal)
    churn_score += num_support_calls * 0.06

    # Electronic check payers churn more (correlates with disengagement)
    churn_score += np.where(payment_method == "Electronic check", 0.10, 0)

    # Senior citizens churn slightly more
    churn_score += np.where(senior_citizen == 1, 0.05, 0)

    # Add random noise so it isn't a deterministic rule (real-world messiness)
    churn_score += rng.normal(0, 0.12, size=n)

    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.68)))  # squash to 0-1
    churn = (rng.random(n) < churn_prob).astype(int)
    churn_labels = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "churn": churn_labels,
        }
    )

    # ------------------------------------------------------------------
    # Inject a small amount of realistic messiness (missing values)
    # so preprocessing.py has real work to do
    # ------------------------------------------------------------------
    missing_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_idx, "total_charges"] = np.nan

    return df


if __name__ == "__main__":
    ensure_dirs()
    print_section("Generating synthetic customer churn dataset")

    df = generate_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")
    print(f"Churn rate: {(df['churn'] == 'Yes').mean():.2%}")
    print("\nSample rows:")
    print(df.head())
