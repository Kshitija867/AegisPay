from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable


from data_loader import PaymentOption, Request
from financial_state import FinancialState, reconstruct_state
from forecasting import (
    check_payment_safety,
    earliest_safe_full_payment_date,
    max_safe_payment_today,
)


MONEY_QUANT = Decimal("0.01")


@dataclass(frozen=True)
class PlanCandidate:
    candidate_id: str
    method: str
    payment_plan: str
    payments: tuple[tuple[date, Decimal], ...]
    total_paid: Decimal
    completion_date: date | None
    spending_changes: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class PlannedDecision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


def _get_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _money(value) -> Decimal:
    if value is None:
        return Decimal("0.00")

    return Decimal(str(value)).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def _decimal(value) -> Decimal | None:
    if value is None or value == "":
        return None

    try:
        return _money(value)
    except Exception:
        return None


def _as_date(value) -> date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    text = str(value).strip()

    if not text:
        return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _request_id(request: Request) -> str:
    return str(
        _get_attr(
            request,
            "request_id",
            default="",
        )
    )


def _request_date(request: Request) -> date:
    value = _get_attr(
        request,
        "request_date",
        "date",
    )

    parsed = _as_date(value)

    if parsed is None:
        raise ValueError(
            f"Request {_request_id(request)} has no valid request date."
        )

    return parsed


def _requested_amount(request: Request) -> Decimal:
    value = _get_attr(
        request,
        "requested_amount",
        "amount",
    )

    parsed = _decimal(value)

    if parsed is None:
        raise ValueError(
            f"Request {_request_id(request)} has no requested amount."
        )

    return parsed


def _partial_allowed(request: Request) -> bool:
    return bool(
        _get_attr(
            request,
            "allows_partial_payment",
            "partial_payment_allowed",
            "allows_partial",
            default=False,
        )
    )


def _deadline(request: Request) -> date | None:
    return _as_date(
        _get_attr(
            request,
            "desired_completion_date",
            "completion_date",
            "desired_payment_date",
            "deadline",
        )
    )


def _user_considers(
    state: FinancialState,
    method: str,
) -> bool:
    allowed = state.payment_methods_user_will_consider

    if allowed is None:
        return True

    return method in allowed


def _option_method(option: PaymentOption) -> str:
    value = _get_attr(
        option,
        "payment_method",
        "method",
        "option_type",
        default="",
    )

    value = str(value).strip().lower()

    if "install" in value:
        return "installments"

    if "full" in value:
        return "full_payment"

    if "partial" in value:
        return "partial_payment"

    return value


def _option_id(option: PaymentOption) -> str:
    value = _get_attr(
        option,
        "payment_option_id",
        "option_id",
        "id",
        default="unknown",
    )

    return str(value)


def _parse_payment_item(item):
    if isinstance(item, dict):

        payment_date = _as_date(
            item.get("date")
            or item.get("payment_date")
            or item.get("due_date")
            or item.get("installment_date")
        )

        amount = _decimal(
            item.get("amount")
            or item.get("payment_amount")
            or item.get("installment_amount")
        )

        if payment_date is not None and amount is not None:
            return payment_date, amount

        return None

    if isinstance(item, (tuple, list)) and len(item) >= 2:

        payment_date = _as_date(item[0])
        amount = _decimal(item[1])

        if payment_date is not None and amount is not None:
            return payment_date, amount

    return None


def _option_payments(
    option: PaymentOption,
    request: Request,
) -> tuple[tuple[date, Decimal], ...]:

    # Representation 1:
    # installments = [(date, amount), ...]
    installments = _get_attr(
        option,
        "installments",
    )

    if installments:

        result = []

        for item in installments:
            parsed = _parse_payment_item(item)

            if parsed is not None:
                result.append(parsed)

        if result:
            return tuple(result)

    # Representation 2:
    # payment_dates + payment_amounts
    dates = _get_attr(
        option,
        "payment_dates",
        "installment_dates",
        "dates",
    )

    amounts = _get_attr(
        option,
        "payment_amounts",
        "installment_amounts",
        "amounts",
    )

    if dates is not None and amounts is not None:

        result = []

        try:
            pairs = zip(dates, amounts)
        except TypeError:
            pairs = []

        for raw_date, raw_amount in pairs:

            payment_date = _as_date(raw_date)
            amount = _decimal(raw_amount)

            if payment_date is not None and amount is not None:
                result.append((payment_date, amount))

        if result:
            return tuple(result)

    # Representation 3:
    # one payment date + one amount
    payment_date = _as_date(
        _get_attr(
            option,
            "payment_date",
            "first_payment_date",
            "start_date",
        )
    )

    amount = _decimal(
        _get_attr(
            option,
            "payment_amount",
            "amount",
            "total_amount",
        )
    )

    if payment_date is not None and amount is not None:
        return ((payment_date, amount),)

    # Full payment fallback.
    if _option_method(option) == "full_payment":

        return (
            (
                _request_date(request),
                _requested_amount(request),
            ),
        )

    return tuple()


def _payment_plan_text(
    payments: Iterable[tuple[date, Decimal]],
) -> str:

    return "|".join(
        f"{payment_date.isoformat()}:{_money(amount):.2f}"
        for payment_date, amount in payments
    )


def _safe_schedule(
    state: FinancialState,
    payments: tuple[tuple[date, Decimal], ...],
) -> bool:

    if not payments:
        return False

    previous_date = None

    for payment_date, amount in payments:

        if amount <= 0:
            return False

        if previous_date is not None:
            if payment_date < previous_date:
                return False

        previous_date = payment_date

        try:
            safety = check_payment_safety(
                state,
                payment_date,
                amount,
            )
        except TypeError:
            try:
                safety = check_payment_safety(
                    state,
                    payment_date,
                    amount,
                )
            except Exception:
                return False
        except Exception:
            return False

        is_safe = getattr(
            safety,
            "is_safe",
            False,
        )

        if not is_safe:
            return False

    return True


def _full_candidate(
    state: FinancialState,
    request: Request,
    option: PaymentOption,
) -> PlanCandidate | None:

    request_date = _request_date(request)
    requested_amount = _requested_amount(request)

    payments = _option_payments(
        option,
        request,
    )

    if len(payments) != 1:
        return None

    payment_date, amount = payments[0]

    if payment_date != request_date:
        return None

    if _money(amount) != _money(requested_amount):
        return None

    if not _safe_schedule(
        state,
        payments,
    ):
        return None

    return PlanCandidate(
        candidate_id=f"full:{_option_id(option)}",
        method="full_payment",
        payment_plan=_payment_plan_text(payments),
        payments=payments,
        total_paid=amount,
        completion_date=payment_date,
        spending_changes=tuple(),
        explanation=(
            "Full payment is safe on the request date while "
            "preserving the required minimum balance."
        ),
    )


def _installment_candidate(
    state: FinancialState,
    request: Request,
    option: PaymentOption,
) -> PlanCandidate | None:

    request_date = _request_date(request)
    deadline = _deadline(request)

    payments = _option_payments(
        option,
        request,
    )

    if len(payments) < 2:
        return None

    # Every installment must occur on/after request date.
    if any(
        payment_date < request_date
        for payment_date, _ in payments
    ):
        return None

    # Must finish by desired completion date.
    if deadline is not None:

        if payments[-1][0] > deadline:
            return None

    # Respect maximum installment months if provided.
    max_months = state.max_installment_months

    if max_months is not None:

        months = (
            (payments[-1][0].year - request_date.year) * 12
            + payments[-1][0].month
            - request_date.month
        )

        if months > max_months:
            return None

    if not _safe_schedule(
        state,
        payments,
    ):
        return None

    total_paid = sum(
        (
            amount
            for _, amount in payments
        ),
        Decimal("0.00"),
    )

    return PlanCandidate(
        candidate_id=f"installments:{_option_id(option)}",
        method="installments",
        payment_plan=_payment_plan_text(payments),
        payments=payments,
        total_paid=total_paid,
        completion_date=payments[-1][0],
        spending_changes=tuple(),
        explanation=(
            "The supplied installment schedule keeps projected "
            "balances above the required minimum and completes "
            "within the allowed payment period."
        ),
    )


def _partial_candidate(
    state: FinancialState,
    request: Request,
) -> PlanCandidate | None:

    if not _partial_allowed(request):
        return None

    if not _user_considers(
        state,
        "partial_payment",
    ):
        return None

    request_date = _request_date(request)
    requested_amount = _requested_amount(request)
    deadline = _deadline(request)

    safe_today = _money(
        max_safe_payment_today(
            state,
            requested_amount,
        )
    )

    if safe_today <= Decimal("0.00"):
        return None

    if safe_today >= requested_amount:
        return None

    full_date = earliest_safe_full_payment_date(
        state,
        requested_amount,
    )

    if full_date is None:
        return None

    if full_date <= request_date:
        return None

    if deadline is not None and full_date > deadline:
        return None

    remainder = _money(
        requested_amount - safe_today
    )

    payments = (
        (request_date, safe_today),
        (full_date, remainder),
    )

    if not _safe_schedule(
        state,
        payments,
    ):
        return None

    return PlanCandidate(
        candidate_id="partial:deterministic",
        method="partial_payment",
        payment_plan=_payment_plan_text(payments),
        payments=payments,
        total_paid=requested_amount,
        completion_date=full_date,
        spending_changes=tuple(),
        explanation=(
            f"Pay {_money(safe_today):.2f} now and the remaining "
            f"{_money(remainder):.2f} on the earliest safe date."
        ),
    )


def _wait_candidate(
    state: FinancialState,
    request: Request,
) -> PlanCandidate | None:

    if not _user_considers(
        state,
        "full_payment",
    ):
        return None

    request_date = _request_date(request)
    requested_amount = _requested_amount(request)
    deadline = _deadline(request)

    safe_date = earliest_safe_full_payment_date(
        state,
        requested_amount,
    )

    if safe_date is None:
        return None

    if safe_date <= request_date:
        return None

    if deadline is not None and safe_date > deadline:
        return None

    payments = (
        (
            safe_date,
            requested_amount,
        ),
    )

    if not _safe_schedule(
        state,
        payments,
    ):
        return None

    return PlanCandidate(
        candidate_id="wait:deterministic",
        method="wait",
        payment_plan=_payment_plan_text(payments),
        payments=payments,
        total_paid=requested_amount,
        completion_date=safe_date,
        spending_changes=tuple(),
        explanation=(
            f"Waiting until {safe_date.isoformat()} allows the "
            "full amount to be paid while preserving the required "
            "minimum balance."
        ),
    )


def _choose_best(
    candidates: list[PlanCandidate],
) -> PlanCandidate | None:

    if not candidates:
        return None

    def ranking(candidate: PlanCandidate):

        completion = (
            candidate.completion_date
            if candidate.completion_date is not None
            else date.max
        )

        return (
            completion,
            len(candidate.spending_changes),
            candidate.total_paid,
            len(candidate.payments),
            candidate.candidate_id,
        )

    return min(
        candidates,
        key=ranking,
    )


def _status(
    candidate: PlanCandidate | None,
) -> str:

    if candidate is None:
        return "not_affordable"

    if candidate.method == "full_payment":
        return "affordable_now"

    if candidate.method in {
        "partial_payment",
        "installments",
    }:
        return "affordable_with_plan"

    if candidate.method == "wait":
        return "affordable_later"

    return "not_affordable"


def plan_request(
    state: FinancialState,
    request: Request,
) -> PlannedDecision:

    request_id = _request_id(request)
    requested_amount = _requested_amount(request)

    safe_today = _money(
        max_safe_payment_today(
            state,
            requested_amount,
        )
    )

    safe_today = max(
        Decimal("0.00"),
        min(
            safe_today,
            requested_amount,
        ),
    )

    earliest = earliest_safe_full_payment_date(
        state,
        requested_amount,
    )

    candidates: list[PlanCandidate] = []

    # Full payment / installments from supplied payment options.
    for option in state.payment_options:

        method = _option_method(option)

        if method == "full_payment":

            if not _user_considers(
                state,
                "full_payment",
            ):
                continue

            candidate = _full_candidate(
                state,
                request,
                option,
            )

        elif method == "installments":

            if not _user_considers(
                state,
                "installments",
            ):
                continue

            candidate = _installment_candidate(
                state,
                request,
                option,
            )

        else:
            candidate = None

        if candidate is not None:
            candidates.append(candidate)

    # Deterministic partial-payment candidate.
    partial = _partial_candidate(
        state,
        request,
    )

    if partial is not None:
        candidates.append(partial)

    # Deterministic wait candidate.
    wait = _wait_candidate(
        state,
        request,
    )

    if wait is not None:
        candidates.append(wait)

    best = _choose_best(candidates)

    if best is None:

        method = "not_recommended"
        affordability = "not_affordable"
        payment_plan = "none"
        spending_changes = "none"

        explanation = (
            f"The requested amount of {_money(requested_amount):.2f} "
            "cannot be safely paid under the available payment "
            "options while preserving the required minimum balance."
        )

    else:

        method = best.method
        affordability = _status(best)
        payment_plan = best.payment_plan

        if best.spending_changes:
            spending_changes = "|".join(
                best.spending_changes
            )
        else:
            spending_changes = "none"

        explanation = best.explanation

    return PlannedDecision(
        request_id=request_id,
        amount_safe_to_pay=safe_today,
        affordability_status=affordability,
        recommended_payment_method=method,
        payment_plan=payment_plan,
        earliest_date_for_full_payment=(
            earliest.isoformat()
            if earliest is not None
            else ""
        ),
        spending_changes_needed=spending_changes,
        decision_explanation=explanation,
    )


def plan_all(
    states: dict[str, FinancialState],
    requests: Iterable[Request],
) -> list[PlannedDecision]:

    decisions = []

    for request in requests:

        request_id = _request_id(request)

        if request_id not in states:
            raise KeyError(
                f"No FinancialState found for {request_id}"
            )

        decisions.append(
            plan_request(
                states[request_id],
                request,
            )
        )

    return decisions


def _run_planner_validation(
    states: dict[str, FinancialState],
    requests: Iterable[Request],
) -> None:

    request_list = list(requests)

    decisions = plan_all(
        states,
        request_list,
    )

    assert len(decisions) == len(request_list)

    request_lookup = {
        _request_id(request): request
        for request in request_list
    }

    for decision in decisions:

        assert decision.request_id

        assert (
            decision.request_id
            in request_lookup
        )

        request = request_lookup[
            decision.request_id
        ]

        requested_amount = _requested_amount(
            request
        )

        assert (
            Decimal("0.00")
            <= decision.amount_safe_to_pay
            <= requested_amount
        )

        assert decision.affordability_status in {
            "affordable_now",
            "affordable_with_plan",
            "affordable_later",
            "not_affordable",
        }

        assert decision.recommended_payment_method in {
            "full_payment",
            "partial_payment",
            "installments",
            "wait",
            "not_recommended",
        }

        assert isinstance(
            decision.payment_plan,
            str,
        )

        assert isinstance(
            decision.decision_explanation,
            str,
        )

    print(
        f"Planner validated successfully for "
        f"{len(decisions)} requests."
    )


if __name__ == "__main__":

    from pathlib import Path

    CODE_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CODE_DIR.parent
    DATASET_DIR = ROOT_DIR / "dataset"

    print("Loading dataset...")

    from data_loader import load_dataset

    dataset = load_dataset(
        DATASET_DIR
    )

    requests = list(
        dataset.requests_by_id.values()
    )

    print(
        f"Loaded {len(requests)} requests, "
        f"{sum(len(v) for v in dataset.payment_options_by_request.values())} payment options."
    )

    print(
        "Building financial states..."
    )

    states = {}

    for request in requests:

        states[
            request.request_id
        ] = reconstruct_state(
            dataset,
            request.request_id,
        )

    print(
        f"Built {len(states)} financial states."
    )

    print(
        "Testing planner across all requests..."
    )

    _run_planner_validation(
        states,
        requests,
    )