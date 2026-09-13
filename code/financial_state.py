"""Deterministic, explainable financial-state reconstruction.

This module classifies supplied records for a single request.  It deliberately
does not forecast balances, select payment plans, or make affordability
decisions; later stages consume the reconstructed state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Iterable

try:  # Supports both ``python code/financial_state.py`` and package imports.
    from .data_loader import Dataset, FinancialEvent, ImageReference, Message, PaymentOption, Request, load_dataset
except ImportError:  # pragma: no cover - exercised by the command-line entry point.
    from data_loader import Dataset, FinancialEvent, ImageReference, Message, PaymentOption, Request, load_dataset


NON_CONFIRMED_CREDIT_TERMS = ("bonus", "commission", "refund", "lottery", "investment gain")
EXCLUDED_STATUSES = frozenset({"cancelled", "failed", "unrealized"})


@dataclass(frozen=True)
class ConvertedEvent:
    """An event plus its optional fixed-rate home-currency representation."""

    event: FinancialEvent
    accounting_date: date | None
    amount_in_home_currency: Decimal | None
    conversion_rate: Decimal | None
    conversion_status: str


@dataclass(frozen=True)
class RecurringSeries:
    """A conservative recurrence inferred from at least three settled events."""

    direction: str
    category: str | None
    description_key: str
    source_event_ids: tuple[str, ...]
    cadence_days: int
    typical_amount_in_home_currency: Decimal
    last_settlement_date: date
    flexibility: str | None
    can_reduce: bool
    can_stop: bool


@dataclass(frozen=True)
class FutureFlexibleExpense:
    """A real future event that a later planner may reference by ``event_id``."""

    event_id: str
    accounting_date: date
    amount: Decimal | None
    currency: str | None
    amount_in_home_currency: Decimal | None
    conversion_rate: Decimal | None
    conversion_status: str
    category: str | None
    description: str | None
    flexibility: str | None
    minimum_allowed_amount: Decimal | None
    minimum_allowed_amount_in_home_currency: Decimal | None
    can_reduce: bool
    can_stop: bool


@dataclass(frozen=True)
class FinancialState:
    request_id: str
    user_id: str
    request_date: date
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...] | None
    protected_expense_categories: tuple[str, ...] | None
    reducible_expense_categories: tuple[str, ...] | None
    stoppable_expense_categories: tuple[str, ...] | None
    payment_methods_user_will_consider: tuple[str, ...] | None
    max_installment_months: int | None
    historical_events: tuple[ConvertedEvent, ...]
    future_events: tuple[ConvertedEvent, ...]
    recurring_income: tuple[RecurringSeries, ...]
    recurring_expenses: tuple[RecurringSeries, ...]
    flexible_future_expenses: tuple[FutureFlexibleExpense, ...]
    confirmed_future_obligations: tuple[ConvertedEvent, ...]
    relevant_messages: tuple[Message, ...]
    relevant_images: tuple[ImageReference, ...]
    payment_options: tuple[PaymentOption, ...]
    conversion_events: tuple[ConvertedEvent, ...]
    excluded_event_ids: tuple[str, ...]


def _accounting_date(event: FinancialEvent) -> date | None:
    return event.settlement_date or event.event_date


def _description_key(event: FinancialEvent) -> str:
    """Retain words while discarding digits that often encode invoice numbers."""
    description = (event.description or "").casefold()
    return " ".join("".join(char for char in word if not char.isdigit()) for word in description.split()).strip()


def _has_non_confirmed_credit_term(event: FinancialEvent) -> bool:
    description = (event.description or "").casefold()
    return any(term in description for term in NON_CONFIRMED_CREDIT_TERMS)


def _sort_events(events: Iterable[FinancialEvent]) -> list[FinancialEvent]:
    return sorted(events, key=lambda event: (_accounting_date(event) or date.min, event.event_id))


class FinancialStateBuilder:
    """Builds request-specific states from the normalized :class:`Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def build_for_request(self, request_id: str) -> FinancialState:
        try:
            request = self.dataset.requests_by_id[request_id]
            profile = self.dataset.profiles_by_user[request.user_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request or profile: {request_id}") from exc

        user_events = _sort_events(self.dataset.events_by_user.get(request.user_id, ()))
        superseded = self._superseded_event_ids(user_events)
        excluded = {
            event.event_id
            for event in user_events
            if event.status in EXCLUDED_STATUSES or event.event_id in superseded
        }
        usable_events = [event for event in user_events if event.event_id not in excluded]
        converted = [self._convert_event(event, profile.home_currency) for event in usable_events]

        historical = tuple(
            entry for entry in converted
            if entry.event.status == "settled" and entry.accounting_date is not None and entry.accounting_date <= request.request_date
        )
        future = tuple(
            entry for entry in converted
            if entry.accounting_date is not None and entry.accounting_date > request.request_date
            and self._is_countable_future_event(entry.event)
        )
        obligations = tuple(
            entry for entry in converted
            if entry.event.direction == "debit" and entry.event.status in {"pending", "scheduled"}
            and (entry.event.status == "pending" or entry.accounting_date is None or entry.accounting_date >= request.request_date)
        )
        historical_settled = [entry for entry in historical if entry.event.status == "settled"]
        recurring_income = self._detect_recurrence(historical_settled, "credit", profile, request.request_date)
        recurring_expenses = self._detect_recurrence(historical_settled, "debit", profile, request.request_date)
        flexible_expenses = tuple(
            expense
            for entry in converted
            if (expense := self._future_flexible_expense(entry, profile, request.request_date)) is not None
        )
        messages = self._relevant_messages(request)
        images = self._relevant_images(request)
        conversions = tuple(entry for entry in converted if entry.event.currency != profile.home_currency)

        return FinancialState(
            request_id=request.request_id, user_id=request.user_id, request_date=request.request_date,
            home_currency=profile.home_currency, current_available_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep, financial_priorities=profile.financial_priorities,
            protected_expense_categories=profile.expense_categories_to_protect,
            reducible_expense_categories=profile.expense_categories_user_is_willing_to_reduce,
            stoppable_expense_categories=profile.expense_categories_user_is_willing_to_stop,
            payment_methods_user_will_consider=profile.payment_methods_user_will_consider,
            max_installment_months=profile.max_installment_months, historical_events=historical,
            future_events=future, recurring_income=recurring_income, recurring_expenses=recurring_expenses,
            flexible_future_expenses=flexible_expenses, confirmed_future_obligations=obligations,
            relevant_messages=messages, relevant_images=images,
            payment_options=self.dataset.payment_options_by_request.get(request.request_id, ()),
            conversion_events=conversions, excluded_event_ids=tuple(sorted(excluded)),
        )

    @staticmethod
    def _superseded_event_ids(events: Iterable[FinancialEvent]) -> set[str]:
        """Resolve only explicit linked replacements; never infer a cancellation."""
        by_id = {event.event_id: event for event in events}
        superseded: set[str] = set()
        for event in events:
            if event.linked_event_id is None:
                continue
            previous = by_id.get(event.linked_event_id)
            if event.status == "cancelled":
                superseded.add(event.linked_event_id)
            elif (
                event.status == "settled"
                and previous is not None
                and previous.status in {"pending", "scheduled"}
                and previous.direction == event.direction
            ):
                # A linked settlement is the explicit final version of an open debit/credit.
                superseded.add(previous.event_id)
        return superseded

    def _convert_event(self, event: FinancialEvent, home_currency: str) -> ConvertedEvent:
        accounting_date = _accounting_date(event)
        if event.amount is None:
            return ConvertedEvent(event, accounting_date, None, None, "amount_missing")
        if event.currency is None:
            return ConvertedEvent(event, accounting_date, None, None, "currency_missing")
        if event.currency == home_currency:
            return ConvertedEvent(event, accounting_date, event.amount, Decimal("1"), "home_currency")
        if accounting_date is None:
            return ConvertedEvent(event, None, None, None, "rate_date_missing")
        rate = self.dataset.exchange_rates_by_key.get((accounting_date, event.currency, home_currency))
        if rate is None:
            return ConvertedEvent(event, accounting_date, None, None, "rate_missing")
        return ConvertedEvent(event, accounting_date, event.amount * rate.rate, rate.rate, "converted")

    @staticmethod
    def _is_countable_future_event(event: FinancialEvent) -> bool:
        """Exclude uncertain credits before a later forecast stage sees them."""
        if event.status == "pending":
            return event.direction == "debit"
        if event.status == "scheduled" and event.direction == "credit":
            return event.event_type == "income" and not _has_non_confirmed_credit_term(event)
        return event.status == "settled"

    @staticmethod
    def _future_flexible_expense(
        entry: ConvertedEvent, profile: object, request_date: date
    ) -> FutureFlexibleExpense | None:
        """Return an eligible supplied future event; never create one from recurrence."""
        event = entry.event
        if (
            entry.accounting_date is None
            or entry.accounting_date < request_date
            or event.status not in {"settled", "scheduled"}
            or event.direction != "debit"
            or event.event_type not in {"expense", "subscription", "debt_payment"}
            or event.category is None
            or event.category in (profile.expense_categories_to_protect or ())
        ):
            return None
        can_reduce = (
            event.flexibility in {"reducible", "reducible_or_stoppable"}
            and event.category in (profile.expense_categories_user_is_willing_to_reduce or ())
        )
        can_stop = (
            event.flexibility in {"stoppable", "reducible_or_stoppable"}
            and event.category in (profile.expense_categories_user_is_willing_to_stop or ())
        )
        if not (can_reduce or can_stop):
            return None
        minimum_home = None
        if event.minimum_allowed_amount is not None and entry.conversion_rate is not None:
            minimum_home = event.minimum_allowed_amount * entry.conversion_rate
        return FutureFlexibleExpense(
            event_id=event.event_id, accounting_date=entry.accounting_date, amount=event.amount,
            currency=event.currency, amount_in_home_currency=entry.amount_in_home_currency,
            conversion_rate=entry.conversion_rate, conversion_status=entry.conversion_status,
            category=event.category, description=event.description, flexibility=event.flexibility,
            minimum_allowed_amount=event.minimum_allowed_amount,
            minimum_allowed_amount_in_home_currency=minimum_home,
            can_reduce=can_reduce, can_stop=can_stop,
        )

    def _detect_recurrence(
        self,
        historical: Iterable[ConvertedEvent],
        direction: str,
        profile: object,
        request_date: date,
    ) -> tuple[RecurringSeries, ...]:
        groups: dict[tuple[str | None, str], list[ConvertedEvent]] = defaultdict(list)
        for entry in historical:
            event = entry.event
            if event.direction != direction or entry.amount_in_home_currency is None or entry.accounting_date is None:
                continue
            if direction == "credit" and (event.event_type != "income" or _has_non_confirmed_credit_term(event)):
                continue
            if direction == "debit" and event.event_type not in {"expense", "subscription", "debt_payment"}:
                continue
            description = _description_key(event)
            if event.category is None or not description:
                continue
            groups[(event.category, description)].append(entry)

        series: list[RecurringSeries] = []
        for (category, description), entries in groups.items():
            entries.sort(key=lambda entry: (entry.accounting_date, entry.event.event_id))
            if len(entries) < 3:
                continue
            dates = [entry.accounting_date for entry in entries if entry.accounting_date is not None]
            intervals = [(later - earlier).days for earlier, later in zip(dates, dates[1:])]
            typical_interval = int(median(intervals))
            if typical_interval < 7 or typical_interval > 370:
                continue
            if max(intervals) - min(intervals) > max(3, round(typical_interval * 0.20)):
                continue
            amounts = [entry.amount_in_home_currency for entry in entries if entry.amount_in_home_currency is not None]
            typical_amount = sum(amounts, Decimal("0")) / len(amounts)
            if typical_amount <= 0 or any(abs(amount - typical_amount) > typical_amount * Decimal("0.10") for amount in amounts):
                continue
            template = entries[-1].event
            protected = category in (profile.expense_categories_to_protect or ())
            flexibility = template.flexibility
            can_reduce = direction == "debit" and not protected and flexibility in {"reducible", "reducible_or_stoppable"} and category in (profile.expense_categories_user_is_willing_to_reduce or ())
            can_stop = direction == "debit" and not protected and flexibility in {"stoppable", "reducible_or_stoppable"} and category in (profile.expense_categories_user_is_willing_to_stop or ())
            series.append(RecurringSeries(direction, category, description, tuple(entry.event.event_id for entry in entries), typical_interval, typical_amount, dates[-1], flexibility, can_reduce, can_stop))
        return tuple(sorted(series, key=lambda item: (item.direction, item.category or "", item.description_key)))

    def _relevant_messages(self, request: Request) -> tuple[Message, ...]:
        records = {message.message_id: message for message in self.dataset.messages_by_user.get(request.user_id, ())}
        records.update({message.message_id: message for message in self.dataset.messages_by_request.get(request.request_id, ())})
        return tuple(sorted(records.values(), key=lambda message: (message.sent_at or datetime.min, message.message_id)))

    def _relevant_images(self, request: Request) -> tuple[ImageReference, ...]:
        records = {image.image_id: image for image in self.dataset.images_by_user.get(request.user_id, ())}
        records.update({image.image_id: image for image in self.dataset.images_by_request.get(request.request_id, ())})
        return tuple(sorted(records.values(), key=lambda image: image.image_id))


def reconstruct_state(dataset: Dataset, request_id: str) -> FinancialState:
    """Convenience API for one request's deterministic financial state."""
    return FinancialStateBuilder(dataset).build_for_request(request_id)


def validate_reconstruction(dataset: Dataset) -> None:
    """Small deterministic integration validation for the complete dataset."""
    states = [reconstruct_state(dataset, request_id) for request_id in sorted(dataset.requests_by_id)]
    assert len(states) == 250
    assert all(isinstance(state.current_available_balance, Decimal) for state in states)
    assert all(isinstance(state.minimum_balance_to_keep, Decimal) for state in states)
    represented_events = [
        entry
        for state in states
        for entry in state.historical_events + state.future_events + state.confirmed_future_obligations
    ]
    assert any(entry.event.amount is None and entry.amount_in_home_currency is None for entry in represented_events)
    assert all(entry.event.status not in EXCLUDED_STATUSES for entry in represented_events)
    assert all(
        entry.amount_in_home_currency is None or isinstance(entry.amount_in_home_currency, Decimal)
        for entry in represented_events
    )
    for state in states:
        obligation_ids = {entry.event.event_id for entry in state.confirmed_future_obligations}
        for event in dataset.events_by_user.get(state.user_id, ()):
            if event.status == "pending" and event.direction == "debit" and event.event_id not in state.excluded_event_ids:
                assert event.event_id in obligation_ids
            if (
                event.status == "scheduled"
                and event.direction == "debit"
                and (_accounting_date(event) is None or _accounting_date(event) >= state.request_date)
                and event.event_id not in state.excluded_event_ids
            ):
                assert event.event_id in obligation_ids
    assert all(not (entry.event.status == "pending" and entry.event.direction == "credit") for state in states for entry in state.future_events)
    assert all(state.payment_options == dataset.payment_options_by_request.get(state.request_id, ()) for state in states)
    assert any(state.relevant_messages for state in states)
    assert any(state.relevant_images for state in states)
    for state in states:
        for expense in state.flexible_future_expenses:
            event = dataset.events_by_id[expense.event_id]
            assert event.user_id == state.user_id
            assert _accounting_date(event) is not None and _accounting_date(event) >= state.request_date
            assert event.status in {"settled", "scheduled"}
            assert event.direction == "debit" and event.flexibility != "fixed"
            assert event.category not in (state.protected_expense_categories or ())
            assert expense.can_reduce or expense.can_stop
            assert expense.amount is None or isinstance(expense.amount, Decimal)
            assert expense.amount_in_home_currency is None or isinstance(expense.amount_in_home_currency, Decimal)


if __name__ == "__main__":
    loaded_dataset = load_dataset(Path(__file__).resolve().parents[1] / "dataset")
    validate_reconstruction(loaded_dataset)
    print(f"Reconstructed and validated {len(loaded_dataset.requests_by_id)} financial states.")
