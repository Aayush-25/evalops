from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from api.evaluator import evaluate_pending_traces

FAKE_ROWS = [
    ("uuid-1", "What is the vacation policy?", "You get 20 days off", ["Policy doc excerpt"]),
    ("uuid-2", "What is the sick leave policy?", "You get 10 days", None),
]


@pytest.fixture
def db_mock():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def get_conn():
        yield conn

    return conn, cursor, get_conn


def _make_df_mock(faith_vals, rel_vals):
    """Return a mock that satisfies df["col"].iloc[i] → float."""
    def make_col(values):
        col = MagicMock()
        col.iloc.__getitem__ = MagicMock(side_effect=lambda i: values[i])
        return col

    df = MagicMock()
    df.__getitem__ = MagicMock(side_effect=lambda col: (
        make_col(faith_vals) if col == "faithfulness" else make_col(rel_vals)
    ))
    return df


class TestEvaluatePendingTraces:
    def test_returns_zero_when_no_pending(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = []
        with patch("database.get_connection", get_conn):
            result = evaluate_pending_traces("test-project")
        assert result == {"evaluated": 0, "errors": 0}

    def test_marks_traces_as_running_before_evaluating(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = FAKE_ROWS
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = _make_df_mock([0.9, 0.8], [0.85, 0.75])

        with patch("database.get_connection", get_conn), \
             patch("api.evaluator.evaluate", return_value=mock_result), \
             patch("api.evaluator.Dataset") as mock_ds:
            mock_ds.from_dict.return_value = MagicMock()
            evaluate_pending_traces("test-project")

        all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
        assert "running" in all_sql

    def test_writes_scores_back_and_returns_count(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = FAKE_ROWS
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = _make_df_mock([0.9, 0.8], [0.85, 0.75])

        with patch("database.get_connection", get_conn), \
             patch("api.evaluator.evaluate", return_value=mock_result), \
             patch("api.evaluator.Dataset") as mock_ds:
            mock_ds.from_dict.return_value = MagicMock()
            result = evaluate_pending_traces("test-project")

        assert result == {"evaluated": 2, "errors": 0}
        all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
        assert "complete" in all_sql

    def test_resets_to_failed_on_ragas_exception(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = FAKE_ROWS

        with patch("database.get_connection", get_conn), \
             patch("api.evaluator.evaluate", side_effect=RuntimeError("RAGAS error")), \
             patch("api.evaluator.Dataset") as mock_ds:
            mock_ds.from_dict.return_value = MagicMock()
            result = evaluate_pending_traces("test-project")

        assert result["evaluated"] == 0
        all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
        assert "failed" in all_sql

    def test_substitutes_empty_string_for_null_contexts(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = [
            ("uuid-1", "question?", "answer", None),
        ]
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = _make_df_mock([0.9], [0.85])

        with patch("database.get_connection", get_conn), \
             patch("api.evaluator.evaluate", return_value=mock_result), \
             patch("api.evaluator.Dataset") as mock_ds:
            mock_ds.from_dict.return_value = MagicMock()
            evaluate_pending_traces("test-project")

        dataset_arg = mock_ds.from_dict.call_args[0][0]
        assert dataset_arg["contexts"] == [[""]]

    def test_builds_dataset_with_correct_structure(self, db_mock):
        _, cursor, get_conn = db_mock
        cursor.fetchall.return_value = FAKE_ROWS
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = _make_df_mock([0.9, 0.8], [0.85, 0.75])

        with patch("database.get_connection", get_conn), \
             patch("api.evaluator.evaluate", return_value=mock_result), \
             patch("api.evaluator.Dataset") as mock_ds:
            mock_ds.from_dict.return_value = MagicMock()
            evaluate_pending_traces("test-project")

        call_data = mock_ds.from_dict.call_args[0][0]
        assert call_data["question"] == ["What is the vacation policy?", "What is the sick leave policy?"]
        assert call_data["answer"] == ["You get 20 days off", "You get 10 days"]
        assert call_data["contexts"][0] == ["Policy doc excerpt"]
        assert call_data["contexts"][1] == [""]  # None → [""]
