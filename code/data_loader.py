"""Deterministic loading and normalization for the Buy or Wait? dataset.

This module intentionally performs no financial inference.  It only converts
the CSV inputs into typed records and lookup indexes for later pipeline stages.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class DatasetValidationError(ValueError):
    """Raised when a required dataset value is malformed or ambiguous."""


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_decimal(value: str | None, *, file_name: str, row_number: int, field: str) -> Decimal | None:
    value = _optional_text(value)
    if value is None:
        return None
    try:
        # Monetary amounts may be formatted with grouping commas in future data.
        return Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise DatasetValidationError(
            f"{file_name} row {row_number}: {field} is not a valid number: {value!r}"
        ) from exc


def _parse_int(value: str | None, *, file_name: str, row_number: int, field: str) -> int | None:
    decimal_value = _parse_decimal(value, file_name=file_name, row_number=row_number, field=field)
    if decimal_value is None:
        return None
    if decimal_value != decimal_value.to_integral_value():
        raise DatasetValidationError(f"{file_name} row {row_number}: {field} must be an integer")
    return int(decimal_value)


def _parse_date(value: str | None, *, file_name: str, row_number: int, field: str) -> date | None:
    value = _optional_text(value)
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DatasetValidationError(
            f"{file_name} row {row_number}: {field} is not an ISO date: {value!r}"
        ) from exc


def _parse_datetime(value: str | None, *, file_name: str, row_number: int, field: str) -> datetime | None:
    value = _optional_text(value)
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetValidationError(
            f"{file_name} row {row_number}: {field} is not an ISO datetime: {value!r}"
        ) from exc


def _parse_bool(value: str | None, *, file_name: str, row_number: int, field: str) -> bool | None:
    value = _optional_text(value)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise DatasetValidationError(f"{file_name} row {row_number}: {field} is not a boolean: {value!r}")


def _parse_pipe_list(value: str | None) -> tuple[str, ...] | None:
    value = _optional_text(value)
    if value is None:
        return None
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _read_rows(dataset_dir: Path, file_name: str) -> list[tuple[int, dict[str, str | None]]]:
    path = dataset_dir / file_name
    if not path.is_file():
        raise DatasetValidationError(f"Required dataset file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise DatasetValidationError(f"{file_name} is missing a header row")
        return [(row_number, row) for row_number, row in enumerate(reader, start=2)]


def _required(value: str | None, *, file_name: str, row_number: int, field: str) -> str:
    result = _optional_text(value)
    if result is None:
        raise DatasetValidationError(f"{file_name} row {row_number}: {field} is required")
    return result


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...] | None
    expense_categories_to_protect: tuple[str, ...] | None
    expense_categories_user_is_willing_to_reduce: tuple[str, ...] | None
    expense_categories_user_is_willing_to_stop: tuple[str, ...] | None
    payment_methods_user_will_consider: tuple[str, ...] | None
    max_installment_months: int | None


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str | None
    category: str | None
    direction: str | None
    amount: Decimal | None
    currency: str | None
    event_date: date | None
    settlement_date: date | None
    status: str | None
    linked_event_id: str | None
    flexibility: str | None
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime | None
    source_type: str | None
    message_text: str | None


@dataclass(frozen=True)
class ImageReference:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date | None
    allows_partial_payment: bool | None
    request_text: str | None


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal | None
    number_of_payments: int | None
    first_payment_date: date | None
    payment_frequency_days: int | None
    financing_fee: Decimal | None
    total_payable_amount: Decimal | None


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Dataset:
    profiles_by_user: dict[str, FinancialProfile]
    events_by_id: dict[str, FinancialEvent]
    events_by_user: dict[str, tuple[FinancialEvent, ...]]
    messages_by_id: dict[str, Message]
    messages_by_user: dict[str, tuple[Message, ...]]
    messages_by_request: dict[str, tuple[Message, ...]]
    messages_by_event: dict[str, tuple[Message, ...]]
    images_by_id: dict[str, ImageReference]
    images_by_user: dict[str, tuple[ImageReference, ...]]
    images_by_request: dict[str, tuple[ImageReference, ...]]
    images_by_event: dict[str, tuple[ImageReference, ...]]
    requests_by_id: dict[str, Request]
    requests_by_user: dict[str, tuple[Request, ...]]
    payment_options_by_id: dict[str, PaymentOption]
    payment_options_by_request: dict[str, tuple[PaymentOption, ...]]
    exchange_rates_by_key: dict[tuple[date, str, str], ExchangeRate]


def _index_unique(records: list[T], key: Callable[[T], str | tuple[date, str, str]], label: str) -> dict:
    indexed = {}
    for record in records:
        key_value = key(record)
        if key_value in indexed:
            raise DatasetValidationError(f"Duplicate {label}: {key_value!r}")
        indexed[key_value] = record
    return indexed


def _group(records: list[T], key: Callable[[T], str | None]) -> dict[str, tuple[T, ...]]:
    grouped: dict[str, list[T]] = {}
    for record in records:
        key_value = key(record)
        if key_value is not None:
            grouped.setdefault(key_value, []).append(record)
    return {key_value: tuple(values) for key_value, values in grouped.items()}


def load_dataset(dataset_dir: str | Path) -> Dataset:
    """Load all supported participant-facing input CSVs from ``dataset_dir``."""
    dataset_path = Path(dataset_dir)

    profiles = [
        FinancialProfile(
            user_id=_required(row["user_id"], file_name="financial_profiles.csv", row_number=n, field="user_id"),
            home_currency=_required(row["home_currency"], file_name="financial_profiles.csv", row_number=n, field="home_currency"),
            current_available_balance=_parse_decimal(row["current_available_balance"], file_name="financial_profiles.csv", row_number=n, field="current_available_balance"),
            minimum_balance_to_keep=_parse_decimal(row["minimum_balance_to_keep"], file_name="financial_profiles.csv", row_number=n, field="minimum_balance_to_keep"),
            financial_priorities=_parse_pipe_list(row["financial_priorities"]),
            expense_categories_to_protect=_parse_pipe_list(row["expense_categories_to_protect"]),
            expense_categories_user_is_willing_to_reduce=_parse_pipe_list(row["expense_categories_user_is_willing_to_reduce"]),
            expense_categories_user_is_willing_to_stop=_parse_pipe_list(row["expense_categories_user_is_willing_to_stop"]),
            payment_methods_user_will_consider=_parse_pipe_list(row["payment_methods_user_will_consider"]),
            max_installment_months=_parse_int(row["max_installment_months"], file_name="financial_profiles.csv", row_number=n, field="max_installment_months"),
        )
        for n, row in _read_rows(dataset_path, "financial_profiles.csv")
    ]
    for profile in profiles:
        if profile.current_available_balance is None or profile.minimum_balance_to_keep is None:
            raise DatasetValidationError(f"financial_profiles.csv: monetary balances are required for {profile.user_id}")

    events = [
        FinancialEvent(
            event_id=_required(row["event_id"], file_name="financial_events.csv", row_number=n, field="event_id"),
            user_id=_required(row["user_id"], file_name="financial_events.csv", row_number=n, field="user_id"),
            event_type=_required(row["event_type"], file_name="financial_events.csv", row_number=n, field="event_type"),
            description=_optional_text(row["description"]), category=_optional_text(row["category"]), direction=_optional_text(row["direction"]),
            amount=_parse_decimal(row["amount"], file_name="financial_events.csv", row_number=n, field="amount"),
            currency=_optional_text(row["currency"]), event_date=_parse_date(row["event_date"], file_name="financial_events.csv", row_number=n, field="event_date"),
            settlement_date=_parse_date(row["settlement_date"], file_name="financial_events.csv", row_number=n, field="settlement_date"),
            status=_optional_text(row["status"]), linked_event_id=_optional_text(row["linked_event_id"]), flexibility=_optional_text(row["flexibility"]),
            minimum_allowed_amount=_parse_decimal(row["minimum_allowed_amount"], file_name="financial_events.csv", row_number=n, field="minimum_allowed_amount"),
        )
        for n, row in _read_rows(dataset_path, "financial_events.csv")
    ]

    messages = [
        Message(_required(row["message_id"], file_name="messages.csv", row_number=n, field="message_id"), _required(row["user_id"], file_name="messages.csv", row_number=n, field="user_id"), _optional_text(row["request_id"]), _optional_text(row["related_event_id"]), _parse_datetime(row["sent_at"], file_name="messages.csv", row_number=n, field="sent_at"), _optional_text(row["source_type"]), _optional_text(row["message_text"]))
        for n, row in _read_rows(dataset_path, "messages.csv")
    ]
    images = [
        ImageReference(_required(row["image_id"], file_name="images.csv", row_number=n, field="image_id"), _required(row["user_id"], file_name="images.csv", row_number=n, field="user_id"), _optional_text(row["request_id"]), _optional_text(row["related_event_id"]))
        for n, row in _read_rows(dataset_path, "images.csv")
    ]
    requests = [
        Request(_required(row["request_id"], file_name="requests.csv", row_number=n, field="request_id"), _required(row["user_id"], file_name="requests.csv", row_number=n, field="user_id"), _parse_date(row["request_date"], file_name="requests.csv", row_number=n, field="request_date"), _required(row["request_type"], file_name="requests.csv", row_number=n, field="request_type"), _parse_decimal(row["requested_amount"], file_name="requests.csv", row_number=n, field="requested_amount"), _parse_date(row["desired_completion_date"], file_name="requests.csv", row_number=n, field="desired_completion_date"), _parse_bool(row["allows_partial_payment"], file_name="requests.csv", row_number=n, field="allows_partial_payment"), _optional_text(row["request_text"]))
        for n, row in _read_rows(dataset_path, "requests.csv")
    ]
    for request in requests:
        if request.request_date is None or request.requested_amount is None:
            raise DatasetValidationError(f"requests.csv: request_date and requested_amount are required for {request.request_id}")

    payment_options = [
        PaymentOption(_required(row["payment_option_id"], file_name="request_payment_options.csv", row_number=n, field="payment_option_id"), _required(row["request_id"], file_name="request_payment_options.csv", row_number=n, field="request_id"), _required(row["payment_method"], file_name="request_payment_options.csv", row_number=n, field="payment_method"), _parse_decimal(row["payment_amount"], file_name="request_payment_options.csv", row_number=n, field="payment_amount"), _parse_int(row["number_of_payments"], file_name="request_payment_options.csv", row_number=n, field="number_of_payments"), _parse_date(row["first_payment_date"], file_name="request_payment_options.csv", row_number=n, field="first_payment_date"), _parse_int(row["payment_frequency_days"], file_name="request_payment_options.csv", row_number=n, field="payment_frequency_days"), _parse_decimal(row["financing_fee"], file_name="request_payment_options.csv", row_number=n, field="financing_fee"), _parse_decimal(row["total_payable_amount"], file_name="request_payment_options.csv", row_number=n, field="total_payable_amount"))
        for n, row in _read_rows(dataset_path, "request_payment_options.csv")
    ]
    exchange_rates = [
        ExchangeRate(_parse_date(row["rate_date"], file_name="exchange_rates.csv", row_number=n, field="rate_date"), _required(row["from_currency"], file_name="exchange_rates.csv", row_number=n, field="from_currency"), _required(row["to_currency"], file_name="exchange_rates.csv", row_number=n, field="to_currency"), _parse_decimal(row["rate"], file_name="exchange_rates.csv", row_number=n, field="rate"))
        for n, row in _read_rows(dataset_path, "exchange_rates.csv")
    ]
    for rate in exchange_rates:
        if rate.rate_date is None or rate.rate is None:
            raise DatasetValidationError("exchange_rates.csv: rate_date and rate are required")

    return Dataset(
        profiles_by_user=_index_unique(profiles, lambda record: record.user_id, "profile user_id"),
        events_by_id=_index_unique(events, lambda record: record.event_id, "event_id"), events_by_user=_group(events, lambda record: record.user_id),
        messages_by_id=_index_unique(messages, lambda record: record.message_id, "message_id"), messages_by_user=_group(messages, lambda record: record.user_id), messages_by_request=_group(messages, lambda record: record.request_id), messages_by_event=_group(messages, lambda record: record.related_event_id),
        images_by_id=_index_unique(images, lambda record: record.image_id, "image_id"), images_by_user=_group(images, lambda record: record.user_id), images_by_request=_group(images, lambda record: record.request_id), images_by_event=_group(images, lambda record: record.related_event_id),
        requests_by_id=_index_unique(requests, lambda record: record.request_id, "request_id"), requests_by_user=_group(requests, lambda record: record.user_id),
        payment_options_by_id=_index_unique(payment_options, lambda record: record.payment_option_id, "payment_option_id"), payment_options_by_request=_group(payment_options, lambda record: record.request_id),
        exchange_rates_by_key=_index_unique(exchange_rates, lambda record: (record.rate_date, record.from_currency, record.to_currency), "exchange-rate key"),
    )


if __name__ == "__main__":
    loaded = load_dataset(Path(__file__).resolve().parents[1] / "dataset")
    print(
        "Loaded "
        f"{len(loaded.profiles_by_user)} profiles, {len(loaded.events_by_id)} events, "
        f"{len(loaded.messages_by_id)} messages, {len(loaded.images_by_id)} images, "
        f"{len(loaded.requests_by_id)} requests, {len(loaded.payment_options_by_id)} payment options, "
        f"and {len(loaded.exchange_rates_by_key)} exchange rates."
    )
