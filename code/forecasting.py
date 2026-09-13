"""Deterministic 90-day financial forecasting and payment safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

try:
    from .financial_state import (
        ConvertedEvent,
        FinancialState,
        FutureFlexibleExpense,
        RecurringSeries,
    )
except ImportError:  # pragma: no cover
    from financial_state import (
        ConvertedEvent,
        FinancialState,
        FutureFlexibleExpense,
        RecurringSeries,
    )


FORECAST_DAYS = 90

# Used only for conservative matching between a projected recurring
# transaction and an actual future event.
AMOUNT_MATCH_TOLERANCE = Decimal("0.01")
DATE_MATCH_TOLERANCE_DAYS = 3


@dataclass(frozen=True)
class ForecastEntry:
    """One financial movement included in the forecast."""

    date: date
    amount: Decimal
    direction: str
    source: str
    event_id: str | None = None


@dataclass(frozen=True)
class DailyBalance:
    """Opening balance, movements, and closing balance for one day."""

    date: date
    opening_balance: Decimal
    credits: Decimal
    debits: Decimal
    closing_balance: Decimal


@dataclass(frozen=True)
class ForecastResult:
    """Complete forecast result."""

    start_date: date
    end_date: date
    minimum_projected_balance: Decimal
    minimum_balance_date: date
    daily_balances: tuple[DailyBalance, ...]
    entries: tuple[ForecastEntry, ...]


@dataclass(frozen=True)
class SafetyCheck:
    """Result of testing a payment against the forecast."""

    safe: bool
    minimum_projected_balance: Decimal
    minimum_balance_date: date
    first_unsafe_date: date | None


def _amounts_match(left: Decimal, right: Decimal) -> bool:
    """Return whether two amounts are close enough to represent one event."""
    return abs(left - right) <= AMOUNT_MATCH_TOLERANCE


def _descriptions_match(
    recurring: RecurringSeries,
    event: ConvertedEvent,
) -> bool:
    """
    Determine whether an actual future event plausibly belongs to a
    recurring series.

    The recurring series was constructed from normalized descriptions,
    category, and historical transactions in financial_state.py.
    """
    event_description = event.event.description or ""

    # Normalize the event description using the same broad approach used
    # by the state reconstruction layer.
    normalized_event = " ".join(event_description.lower().split())

    if normalized_event != recurring.description_key:
        return False

    if recurring.category is not None:
        if event.event.category != recurring.category:
            return False

    return True


def _future_event_matches_recurring_projection(
    event: ConvertedEvent,
    recurring: RecurringSeries,
    projected_date: date,
) -> bool:
    """Check whether an actual future event replaces a recurring projection."""

    if event.accounting_date is None:
        return False

    if event.event.direction != (
        "credit" if recurring.direction == "income" else "debit"
    ):
        return False

    if abs((event.accounting_date - projected_date).days) > DATE_MATCH_TOLERANCE_DAYS:
        return False

    if event.amount_in_home_currency is None:
        return False

    if not _amounts_match(
        event.amount_in_home_currency,
        recurring.typical_amount_in_home_currency,
    ):
        return False

    if not _descriptions_match(recurring, event):
        return False

    return True


def _add_recurring_entries(
    entries: list[ForecastEntry],
    series: tuple[RecurringSeries, ...],
    future_events: tuple[ConvertedEvent, ...],
    start_date: date,
    end_date: date,
) -> None:
    """
    Add conservative recurring projections.

    If the dataset already contains an actual future event corresponding
    to a projected recurring transaction, the projection is skipped so
    that the same transaction is not counted twice.
    """

    for item in series:
        if item.cadence_days <= 0:
            continue

        next_date = item.last_settlement_date + timedelta(
            days=item.cadence_days
        )

        while next_date <= end_date:
            if next_date >= start_date:
                already_supplied = any(
                    _future_event_matches_recurring_projection(
                        event,
                        item,
                        next_date,
                    )
                    for event in future_events
                )

                if not already_supplied:
                    entries.append(
                        ForecastEntry(
                            date=next_date,
                            amount=item.typical_amount_in_home_currency,
                            direction=item.direction,
                            source="recurring",
                        )
                    )

            next_date += timedelta(days=item.cadence_days)


def _event_to_entry(
    event: ConvertedEvent,
    start_date: date,
    end_date: date,
) -> ForecastEntry | None:
    """Convert a state event into a forecast entry."""

    if event.accounting_date is None:
        return None

    if not (start_date <= event.accounting_date <= end_date):
        return None

    if event.amount_in_home_currency is None:
        return None

    if event.event.direction not in {"credit", "debit"}:
        return None

    return ForecastEntry(
        date=event.accounting_date,
        amount=event.amount_in_home_currency,
        direction=event.event.direction,
        source=event.event.status,
        event_id=event.event.event_id,
    )


def _event_to_pending_debit_entry(
    event: ConvertedEvent,
    start_date: date,
    end_date: date,
) -> ForecastEntry | None:
    """
    Convert a pending debit into a forecast entry.

    Pending debits must be reserved even when their original event date
    is stale. If the event's accounting date is before the forecast start,
    treat the obligation as present from the request date.
    """

    if event.event.direction != "debit":
        return None

    if event.amount_in_home_currency is None:
        return None

    accounting_date = event.accounting_date

    if accounting_date is None or accounting_date < start_date:
        accounting_date = start_date

    if accounting_date > end_date:
        return None

    return ForecastEntry(
        date=accounting_date,
        amount=event.amount_in_home_currency,
        direction="debit",
        source="pending",
        event_id=event.event.event_id,
    )


def _build_entries(
    state: FinancialState,
    start_date: date,
    end_date: date,
) -> list[ForecastEntry]:
    """Build all forecast entries without double counting."""

    entries: list[ForecastEntry] = []

    # First collect actual future events.
    actual_future_entries: list[ForecastEntry] = []

    for event in state.future_events:
        entry = _event_to_entry(
            event,
            start_date,
            end_date,
        )

        if entry is not None:
            actual_future_entries.append(entry)

    # Actual future events are authoritative. Recurring projections only
    # fill dates where no corresponding actual future event exists.
    _add_recurring_entries(
        entries=entries,
        series=state.recurring_income,
        future_events=state.future_events,
        start_date=start_date,
        end_date=end_date,
    )

    _add_recurring_entries(
        entries=entries,
        series=state.recurring_expenses,
        future_events=state.future_events,
        start_date=start_date,
        end_date=end_date,
    )

    entries.extend(actual_future_entries)

    # Confirmed future obligations contain scheduled/pending debits that
    # are not necessarily present in state.future_events.
    known_event_ids = {
        entry.event_id
        for entry in entries
        if entry.event_id is not None
    }

    for event in state.confirmed_future_obligations:
        event_id = event.event.event_id

        if event_id in known_event_ids:
            continue

        if event.event.status == "pending":
            entry = _event_to_pending_debit_entry(
                event,
                start_date,
                end_date,
            )
        else:
            entry = _event_to_entry(
                event,
                start_date,
                end_date,
            )

        if entry is None:
            continue

        # Only obligations that are actual debits should affect the
        # payment-safety forecast.
        if entry.direction != "debit":
            continue

        entries.append(entry)
        known_event_ids.add(event_id)

    return sorted(
        entries,
        key=lambda item: (
            item.date,
            item.direction,
            item.event_id or "",
            item.source,
        ),
    )


def forecast(
    state: FinancialState,
    days: int = FORECAST_DAYS,
) -> ForecastResult:
    """
    Build a deterministic daily financial forecast.

    The forecast starts on request_date and covers exactly `days`
    calendar days after the request date, inclusive.
    """

    if days < 0:
        raise ValueError("days must be non-negative")

    start_date = state.request_date
    end_date = start_date + timedelta(days=days)

    entries = _build_entries(
        state,
        start_date,
        end_date,
    )

    by_date: dict[date, list[ForecastEntry]] = {}

    for entry in entries:
        by_date.setdefault(entry.date, []).append(entry)

    daily_balances: list[DailyBalance] = []

    balance = state.current_available_balance

    minimum_balance = balance
    minimum_date = start_date

    current_date = start_date

    while current_date <= end_date:
        opening_balance = balance

        credits = sum(
            (
                entry.amount
                for entry in by_date.get(current_date, ())
                if entry.direction == "credit"
            ),
            Decimal("0"),
        )

        debits = sum(
            (
                entry.amount
                for entry in by_date.get(current_date, ())
                if entry.direction == "debit"
            ),
            Decimal("0"),
        )

        balance = opening_balance + credits - debits

        daily = DailyBalance(
            date=current_date,
            opening_balance=opening_balance,
            credits=credits,
            debits=debits,
            closing_balance=balance,
        )

        daily_balances.append(daily)

        if balance < minimum_balance:
            minimum_balance = balance
            minimum_date = current_date

        current_date += timedelta(days=1)

    return ForecastResult(
        start_date=start_date,
        end_date=end_date,
        minimum_projected_balance=minimum_balance,
        minimum_balance_date=minimum_date,
        daily_balances=tuple(daily_balances),
        entries=tuple(entries),
    )


def check_payment_safety(
    state: FinancialState,
    payment_date: date,
    payment_amount: Decimal,
    days: int = FORECAST_DAYS,
) -> SafetyCheck:
    """
    Check whether a one-time payment is safe.

    The payment is applied on payment_date and remains deducted from
    every subsequent day's balance.

    Every day in the complete forecast must remain at or above the
    user's minimum required balance.
    """

    if payment_amount < 0:
        raise ValueError("payment_amount must be non-negative")

    result = forecast(
        state,
        days=days,
    )

    minimum_balance = result.minimum_projected_balance
    minimum_date = result.minimum_balance_date

    first_unsafe_date: date | None = None

    for daily in result.daily_balances:
        adjusted_balance = daily.closing_balance

        if daily.date >= payment_date:
            adjusted_balance -= payment_amount

        if adjusted_balance < minimum_balance:
            minimum_balance = adjusted_balance
            minimum_date = daily.date

        if (
            daily.date >= payment_date
            and adjusted_balance < state.minimum_balance_to_keep
            and first_unsafe_date is None
        ):
            first_unsafe_date = daily.date

        # A baseline forecast violation before the payment date also means
        # that the proposed payment cannot make the complete forecast safe.
        if (
            daily.date < payment_date
            and daily.closing_balance < state.minimum_balance_to_keep
            and first_unsafe_date is None
        ):
            first_unsafe_date = daily.date

    return SafetyCheck(
        safe=first_unsafe_date is None,
        minimum_projected_balance=minimum_balance,
        minimum_balance_date=minimum_date,
        first_unsafe_date=first_unsafe_date,
    )


def max_safe_payment_today(
    state: FinancialState,
    days: int = FORECAST_DAYS,
) -> Decimal:
    """
    Return the maximum one-time payment that can be made today while
    keeping the entire forecast above the minimum required balance.
    """

    result = forecast(
        state,
        days=days,
    )

    # If the baseline forecast itself violates the minimum balance,
    # there is no safe amount available for an additional payment.
    if any(
        daily.closing_balance < state.minimum_balance_to_keep
        for daily in result.daily_balances
    ):
        return Decimal("0")

    available = (
        result.minimum_projected_balance
        - state.minimum_balance_to_keep
    )

    return max(
        Decimal("0"),
        available,
    )


def earliest_safe_full_payment_date(
    state: FinancialState,
    requested_amount: Decimal,
    days: int = FORECAST_DAYS,
) -> date | None:
    """
    Return the earliest date on which the full requested amount can be
    paid as a single payment while keeping the entire forecast safe.

    Importantly, dates BEFORE the proposed payment are checked too.
    """

    if requested_amount < 0:
        raise ValueError("requested_amount must be non-negative")

    result = forecast(
        state,
        days=days,
    )

    for candidate in result.daily_balances:
        candidate_date = candidate.date

        safe_for_entire_forecast = True

        for daily in result.daily_balances:
            adjusted_balance = daily.closing_balance

            if daily.date >= candidate_date:
                adjusted_balance -= requested_amount

            if adjusted_balance < state.minimum_balance_to_keep:
                safe_for_entire_forecast = False
                break

        if safe_for_entire_forecast:
            return candidate_date

    return None


def validate_forecast(
    state: FinancialState,
) -> ForecastResult:
    """Run structural invariants over a standard 90-day forecast."""

    result = forecast(
        state,
        days=FORECAST_DAYS,
    )

    assert len(result.daily_balances) == FORECAST_DAYS + 1

    assert result.start_date == state.request_date

    assert result.end_date == (
        state.request_date + timedelta(days=FORECAST_DAYS)
    )

    for daily in result.daily_balances:
        assert (
            daily.closing_balance
            == daily.opening_balance
            + daily.credits
            - daily.debits
        )

        assert daily.credits >= 0
        assert daily.debits >= 0

    # Ensure entries are within the forecast window.
    for entry in result.entries:
        assert result.start_date <= entry.date <= result.end_date
        assert entry.amount >= 0
        assert entry.direction in {"credit", "debit"}

    return result


def _run_all_request_validation() -> None:
    """
    Validate forecasting across every request.

    This is deliberately kept lightweight so it can be used as a
    terminal smoke/regression test.
    """

    from pathlib import Path

    try:
        from .data_loader import load_dataset
        from .financial_state import reconstruct_state
    except ImportError:  # pragma: no cover
        from data_loader import load_dataset
        from financial_state import reconstruct_state

    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"

    dataset = load_dataset(dataset_dir)

    request_ids = sorted(dataset.requests_by_id)

    if not request_ids:
        raise RuntimeError("No requests found in dataset")

    validated = 0

    for request_id in request_ids:
        state = reconstruct_state(
            dataset,
            request_id,
        )

        result = validate_forecast(state)

        # Check that the max-safe-payment calculation is deterministic.
        first = max_safe_payment_today(state)
        second = max_safe_payment_today(state)

        assert first == second

        validated += 1

    print(
        f"Forecast validation passed for {validated} requests."
    )


if __name__ == "__main__":
    from pathlib import Path

    try:
        from .data_loader import load_dataset
        from .financial_state import reconstruct_state
    except ImportError:  # pragma: no cover
        from data_loader import load_dataset
        from financial_state import reconstruct_state

    dataset_dir = (
        Path(__file__).resolve().parents[1] / "dataset"
    )

    dataset = load_dataset(dataset_dir)

    first_request_id = sorted(
        dataset.requests_by_id
    )[0]

    state = reconstruct_state(
        dataset,
        first_request_id,
    )

    result = validate_forecast(state)

    print(
        f"Forecast validated for {first_request_id}: "
        f"{result.start_date} -> {result.end_date}"
    )

    print(
        "Minimum projected balance: "
        f"{result.minimum_projected_balance} "
        f"on {result.minimum_balance_date}"
    )

    print(
        "Maximum safe payment today: "
        f"{max_safe_payment_today(state)}"
    )

    print()
    print("Running validation across all requests...")

    _run_all_request_validation()