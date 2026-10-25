"""Tests for EvalReportSink implementations."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

from michelangelo.gen.api.v2.chart_pb2 import (
    Chart,
    ChartDataPoint,
    ChartDataset,
    ChartType,
)
from michelangelo.gen.api.v2.evaluation_report_pb2 import (
    EvaluationReport,
    EvaluationReportSpec,
)
from michelangelo.workflow.schema.eval_report_sinks.local_file import (
    LocalFileEvalReportSinkConfig,
)
from michelangelo.workflow.tasks.functions.eval_report_sinks.base import (
    EvalReportSink,
    flatten_report_to_metric_history,
    flatten_report_to_metrics,
)
from michelangelo.workflow.tasks.functions.eval_report_sinks.local_file import (
    LocalFileEvalReportSink,
)


def _report(name: str = "test-report", namespace: str = "") -> EvaluationReport:
    """Build a minimal EvaluationReport for test use."""
    r = EvaluationReport(spec=EvaluationReportSpec(title="Test"))
    r.metadata.name = name
    r.metadata.namespace = namespace
    return r


class TestEvalReportSinkABC(TestCase):
    """Tests for the EvalReportSink abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """It raises TypeError when EvalReportSink is instantiated directly."""
        with self.assertRaises(TypeError):
            EvalReportSink()  # type: ignore[abstract]


class TestLocalFileEvalReportSink(TestCase):
    """Tests for LocalFileEvalReportSink."""

    def setUp(self) -> None:
        """Track output dirs for cleanup."""
        self._output_dirs: list[str] = []

    def tearDown(self) -> None:
        """Remove temp dirs created during tests."""
        for d in self._output_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def _write(self, report: EvaluationReport, **kwargs: Any):
        sink = LocalFileEvalReportSink(**kwargs)
        result = sink.write(report)
        self._output_dirs.append(os.path.dirname(result.output_path))
        return result

    def test_writes_valid_json_file(self):
        """It writes a valid JSON file and returns the path."""
        result = self._write(_report())
        self.assertTrue(os.path.exists(result.output_path))
        with open(result.output_path) as f:
            doc = json.load(f)
        self.assertIn("spec", doc)

    def test_filename_matches_report_name(self):
        """It names the file after report.metadata.name."""
        result = self._write(_report(name="my-eval"))
        self.assertTrue(result.output_path.endswith("my-eval.json"))

    def test_output_dir_auto_created_when_config_none(self):
        """It creates a michelangelo_reports_ temp dir when no config given."""
        result = self._write(_report())
        parent = os.path.basename(os.path.dirname(result.output_path))
        self.assertTrue(parent.startswith("michelangelo_reports_"))

    def test_explicit_output_dir_used(self):
        """It writes to the configured output_dir."""
        import tempfile

        d = tempfile.mkdtemp()
        self._output_dirs.append(d)
        cfg = LocalFileEvalReportSinkConfig(output_dir=d)
        sink = LocalFileEvalReportSink(cfg)
        result = sink.write(_report())
        self.assertTrue(result.output_path.startswith(d))

    def test_extra_fields_merged_into_json(self):
        """It merges extra_fields into the written JSON document."""
        sink = LocalFileEvalReportSink()
        result = sink.write(_report(), extra_fields={"ci_run": "build-42"})
        self._output_dirs.append(os.path.dirname(result.output_path))
        with open(result.output_path) as f:
            doc = json.load(f)
        self.assertEqual(doc["ci_run"], "build-42")

    def test_result_name_and_namespace(self):
        """It returns the report name and namespace in the result."""
        result = self._write(_report(name="r1", namespace="ns-prod"))
        self.assertEqual(result.name, "r1")
        self.assertEqual(result.namespace, "ns-prod")

    def test_raises_when_name_not_set(self):
        """It raises ValueError when report.metadata.name is empty."""
        report = EvaluationReport(spec=EvaluationReportSpec(title="T"))
        with self.assertRaises(ValueError):
            LocalFileEvalReportSink().write(report)

    def test_snake_case_field_names_in_output(self):
        """It serializes proto fields with snake_case keys."""
        result = self._write(_report())
        with open(result.output_path) as f:
            doc = json.load(f)
        self.assertNotIn("typeMeta", doc)


class TestEvaluationReportService(TestCase):
    """Tests for EvaluationReportService CRUD methods."""

    def _make_service(self):
        """Create an EvaluationReportService with a mocked gRPC stub."""
        from michelangelo.api.v2.services.base import Context, DefaultHeaderProvider
        from michelangelo.api.v2.services.gen.evaluation_report import (
            EvaluationReportService,
        )

        ctx = Context()
        ctx._channel = MagicMock()
        ctx._header_provider = DefaultHeaderProvider()
        ctx._header_provider._caller = "test-caller"
        svc = EvaluationReportService(ctx)
        svc._service_stub = MagicMock()
        return svc

    def test_create_evaluation_report_calls_stub(self):
        """create_evaluation_report passes the report to the stub and returns it."""
        svc = self._make_service()
        report = _report(name="q1-eval", namespace="my-project")
        svc._service_stub.CreateEvaluationReport.return_value = MagicMock(
            evaluation_report=report
        )

        result = svc.create_evaluation_report(report)

        svc._service_stub.CreateEvaluationReport.assert_called_once()
        self.assertEqual(result.metadata.name, "q1-eval")

    def test_get_evaluation_report_passes_namespace_and_name(self):
        """get_evaluation_report builds the correct GetEvaluationReportRequest."""
        svc = self._make_service()
        expected = _report(name="q1-eval", namespace="my-project")
        svc._service_stub.GetEvaluationReport.return_value = MagicMock(
            evaluation_report=expected
        )

        result = svc.get_evaluation_report(namespace="my-project", name="q1-eval")

        req = svc._service_stub.GetEvaluationReport.call_args[0][0]
        self.assertEqual(req.namespace, "my-project")
        self.assertEqual(req.name, "q1-eval")
        self.assertEqual(result.metadata.name, "q1-eval")

    def test_update_evaluation_report_returns_server_response(self):
        """update_evaluation_report returns the server-confirmed proto."""
        svc = self._make_service()
        updated = _report(name="q1-eval", namespace="my-project")
        svc._service_stub.UpdateEvaluationReport.return_value = MagicMock(
            evaluation_report=updated
        )

        result = svc.update_evaluation_report(_report(name="q1-eval"))

        svc._service_stub.UpdateEvaluationReport.assert_called_once()
        self.assertEqual(result.metadata.name, "q1-eval")

    def test_delete_evaluation_report_sends_correct_request(self):
        """delete_evaluation_report builds DeleteEvaluationReportRequest correctly."""
        svc = self._make_service()
        svc._service_stub.DeleteEvaluationReport.return_value = MagicMock()

        svc.delete_evaluation_report(namespace="my-project", name="q1-eval")

        req = svc._service_stub.DeleteEvaluationReport.call_args[0][0]
        self.assertEqual(req.namespace, "my-project")
        self.assertEqual(req.name, "q1-eval")

    def test_delete_evaluation_report_collection_targets_namespace(self):
        """delete_evaluation_report_collection targets the correct namespace."""
        svc = self._make_service()
        svc._service_stub.DeleteEvaluationReportCollection.return_value = MagicMock()

        svc.delete_evaluation_report_collection(namespace="my-project")

        req = svc._service_stub.DeleteEvaluationReportCollection.call_args[0][0]
        self.assertEqual(req.namespace, "my-project")

    def test_list_evaluation_report_returns_list(self):
        """list_evaluation_report returns the EvaluationReportList from the stub."""
        svc = self._make_service()
        mock_list = MagicMock()
        mock_list.items = [_report(name="r1"), _report(name="r2")]
        svc._service_stub.ListEvaluationReport.return_value = MagicMock(
            evaluation_report_list=mock_list
        )

        result = svc.list_evaluation_report(namespace="my-project")

        svc._service_stub.ListEvaluationReport.assert_called_once()
        self.assertEqual(len(result.items), 2)


class TestAPIClientEvalReportSink(TestCase):
    """Tests for APIClientEvalReportSink (delegates to APIClient)."""

    def _make_sink(self, mock_apiclient: MagicMock):
        """Build an APIClientEvalReportSink with a mocked APIClient."""
        from michelangelo.workflow.tasks.functions.eval_report_sinks.api import (
            APIClientEvalReportSink,
        )

        with patch("michelangelo.api.v2.APIClient", mock_apiclient):
            return APIClientEvalReportSink()

    def _make_created(
        self, report_name: str = "api-report", namespace: str = "ns"
    ) -> EvaluationReport:
        created = EvaluationReport()
        created.metadata.name = report_name
        created.metadata.namespace = namespace
        return created

    def test_delegates_to_apiclient_evaluation_report_service(self):
        """It binds _svc to APIClient.EvaluationReportService at construction."""
        mock_svc = MagicMock()
        mock_apiclient = MagicMock()
        mock_apiclient.EvaluationReportService = mock_svc

        sink = self._make_sink(mock_apiclient)

        self.assertIs(sink._svc, mock_svc)

    def test_accepts_injected_svc_for_di(self):
        """It accepts an explicit svc param without touching APIClient."""
        from michelangelo.workflow.tasks.functions.eval_report_sinks.api import (
            APIClientEvalReportSink,
        )

        mock_svc = MagicMock()
        sink = APIClientEvalReportSink(svc=mock_svc)
        self.assertIs(sink._svc, mock_svc)

    def test_raises_when_apiclient_service_is_none(self):
        """It raises RuntimeError when APIClient.EvaluationReportService is None."""
        from michelangelo.workflow.tasks.functions.eval_report_sinks.api import (
            APIClientEvalReportSink,
        )

        mock_apiclient = MagicMock()
        mock_apiclient.EvaluationReportService = None

        with (
            patch("michelangelo.api.v2.APIClient", mock_apiclient),
            self.assertRaises(RuntimeError) as ctx,
        ):
            APIClientEvalReportSink()
        self.assertIn("MA_API_SERVER", str(ctx.exception))

    def test_write_calls_create_evaluation_report(self):
        """write() calls svc.create_evaluation_report with the report."""
        mock_apiclient = MagicMock()
        created = self._make_created("r1", "ns1")
        report = _report(name="r1")
        mock_apiclient.EvaluationReportService.create_evaluation_report.return_value = (
            created
        )

        sink = self._make_sink(mock_apiclient)
        result = sink.write(report)

        mock_apiclient.EvaluationReportService.create_evaluation_report.assert_called_once_with(
            report
        )
        self.assertEqual(result.name, "r1")
        self.assertEqual(result.namespace, "ns1")

    def test_namespace_not_injected(self):
        """write() does not mutate report.metadata.namespace."""
        mock_apiclient = MagicMock()
        created = self._make_created("r1", "caller-ns")
        mock_apiclient.EvaluationReportService.create_evaluation_report.return_value = (
            created
        )

        sink = self._make_sink(mock_apiclient)
        report = _report(name="r1", namespace="caller-ns")
        original_ns = report.metadata.namespace
        sink.write(report)

        self.assertEqual(report.metadata.namespace, original_ns)

    def test_no_channel_owned(self):
        """APIClientEvalReportSink holds no channel reference."""
        mock_apiclient = MagicMock()
        sink = self._make_sink(mock_apiclient)
        self.assertFalse(hasattr(sink, "_channel"))

    def test_extra_fields_emits_user_warning(self):
        """write() emits UserWarning when extra_fields is non-empty."""
        mock_apiclient = MagicMock()
        created = self._make_created("r1", "ns1")
        mock_apiclient.EvaluationReportService.create_evaluation_report.return_value = (
            created
        )

        sink = self._make_sink(mock_apiclient)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sink.write(_report(name="r1"), extra_fields={"key": "value"})

        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
        self.assertTrue(any("extra_fields" in str(w.message) for w in caught))

    def test_extra_fields_none_does_not_warn(self):
        """write() emits no warning when extra_fields is None or omitted."""
        mock_apiclient = MagicMock()
        created = self._make_created("r1", "ns1")
        mock_apiclient.EvaluationReportService.create_evaluation_report.return_value = (
            created
        )

        sink = self._make_sink(mock_apiclient)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sink.write(_report(name="r1"))  # no extra_fields

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(user_warnings, [])

    def test_non_rpc_error_propagates_unchanged(self):
        """Non-gRPC exceptions from write() are not wrapped as OSError."""
        mock_apiclient = MagicMock()
        mock_apiclient.EvaluationReportService.create_evaluation_report.side_effect = (
            ValueError("bad proto")
        )

        sink = self._make_sink(mock_apiclient)
        with self.assertRaises(ValueError) as ctx:
            sink.write(_report(name="r1"))
        self.assertIn("bad proto", str(ctx.exception))

    def test_grpc_rpc_error_raised_as_oserror(self):
        """It wraps grpc.RpcError as OSError."""
        import grpc

        class _FakeRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "server unreachable"

        mock_apiclient = MagicMock()
        mock_apiclient.EvaluationReportService.create_evaluation_report.side_effect = (
            _FakeRpcError()
        )

        sink = self._make_sink(mock_apiclient)
        with self.assertRaises(OSError) as ctx:
            sink.write(_report(name="r1"))
        self.assertIn("APIClientEvalReportSink", str(ctx.exception))


class TestFlattenReportToMetrics(TestCase):
    """Tests for flatten_report_to_metrics() and flatten_report_to_metric_history().

    Reports are built from the real Chart protos so the helpers are checked
    against the schema producers actually emit.
    """

    @staticmethod
    def _report_with(*charts: Chart) -> EvaluationReport:
        r = _report()
        r.spec.charts.extend(charts)
        return r

    @staticmethod
    def _line_chart(title: str, *lines: tuple[str, list[tuple[float, float]]]):
        chart = Chart(chart_type=ChartType.CHART_TYPE_LINE_CHART, title=title)
        for name, points in lines:
            chart.line_chart.lines.append(
                ChartDataset(
                    line_name=name,
                    points=[ChartDataPoint(x=x, y=y) for x, y in points],
                )
            )
        return chart

    def test_empty_report_returns_empty_dict(self):
        """It returns {} when there are no charts."""
        self.assertEqual(flatten_report_to_metrics(_report()), {})
        self.assertEqual(flatten_report_to_metric_history(_report()), {})

    def test_scalar_chart_extracted(self):
        """It extracts title -> number for a scalar chart."""
        chart = Chart(chart_type=ChartType.CHART_TYPE_SCALAR, title="accuracy")
        chart.scalar.cell_value.number = 0.95
        result = flatten_report_to_metrics(self._report_with(chart))
        self.assertEqual(result, {"accuracy": 0.95})

    def test_scalar_name_and_index_fallbacks(self):
        """It uses scalar_name, then metric_<i>, when the title is empty."""
        named = Chart(chart_type=ChartType.CHART_TYPE_SCALAR)
        named.scalar.scalar_name = "auc"
        named.scalar.cell_value.number = 0.7
        unnamed = Chart(chart_type=ChartType.CHART_TYPE_SCALAR)
        unnamed.scalar.cell_value.number = 0.8
        result = flatten_report_to_metrics(self._report_with(named, unnamed))
        self.assertEqual(result, {"auc": 0.7, "metric_1": 0.8})

    def test_scalar_str_value(self):
        """It parses a numeric str_value and drops a non-numeric one."""
        good = Chart(chart_type=ChartType.CHART_TYPE_SCALAR, title="f1")
        good.scalar.cell_value.str_value = "0.5"
        bad = Chart(chart_type=ChartType.CHART_TYPE_SCALAR, title="bad")
        bad.scalar.cell_value.str_value = "n/a"
        result = flatten_report_to_metrics(self._report_with(good, bad))
        self.assertEqual(result, {"f1": 0.5})

    def test_single_point_line_uses_chart_title(self):
        """It maps a one-line, one-point chart to its title."""
        chart = self._line_chart("loss", ("train", [(3.0, 0.25)]))
        result = flatten_report_to_metrics(self._report_with(chart))
        self.assertEqual(result, {"loss": 0.25})

    def test_multi_series_keys_use_line_name(self):
        """It keys each series of a multi-series chart by title/line_name."""
        chart = self._line_chart("loss", ("train", [(1.0, 0.3)]), ("", [(1.0, 0.4)]))
        result = flatten_report_to_metrics(self._report_with(chart))
        self.assertEqual(result, {"loss/train": 0.3, "loss/series_1": 0.4})

    def test_other_dataset_chart_types(self):
        """It reads point, bar and time-series charts the same way."""
        point = Chart(chart_type=ChartType.CHART_TYPE_POINT_CHART, title="p")
        point.point_chart.point_sets.append(
            ChartDataset(points=[ChartDataPoint(x=0, y=1.0)])
        )
        bar = Chart(chart_type=ChartType.CHART_TYPE_BAR_CHART, title="b")
        bar.bar_chart.bars.append(ChartDataset(points=[ChartDataPoint(x=0, y=2.0)]))
        ts = Chart(chart_type=ChartType.CHART_TYPE_TIME_SERIES, title="t")
        ts.time_series.lines.append(ChartDataset(points=[ChartDataPoint(x=0, y=3.0)]))
        result = flatten_report_to_metrics(self._report_with(point, bar, ts))
        self.assertEqual(result, {"p": 1.0, "b": 2.0, "t": 3.0})

    def test_multi_point_series_skipped_with_warning(self):
        """It leaves multi-point series out of the scalar dict and warns."""
        import logging

        chart = self._line_chart("curve", ("val", [(0, 0.9), (1, 0.8)]))
        with self.assertLogs(
            "michelangelo.workflow.tasks.functions.eval_report_sinks.base",
            level=logging.WARNING,
        ):
            result = flatten_report_to_metrics(self._report_with(chart))
        self.assertEqual(result, {})

    def test_history_keeps_every_point_with_its_step(self):
        """It returns (x, y) pairs for every series, including multi-point."""
        curve = self._line_chart(
            "loss", ("train", [(0, 0.9), (1, 0.7)]), ("val", [(0, 1.0)])
        )
        scalar = Chart(chart_type=ChartType.CHART_TYPE_SCALAR, title="acc")
        scalar.scalar.cell_value.number = 0.9
        result = flatten_report_to_metric_history(self._report_with(curve, scalar))
        self.assertEqual(
            result,
            {
                "loss/train": [(0.0, 0.9), (1.0, 0.7)],
                "loss/val": [(0.0, 1.0)],
                "acc": [(0.0, 0.9)],
            },
        )

    def test_unsupported_chart_types_ignored(self):
        """It ignores tables and other charts without numeric series."""
        chart = Chart(chart_type=ChartType.CHART_TYPE_TABLE, title="Metrics")
        chart.table.column_names.append("metric")
        self.assertEqual(flatten_report_to_metrics(self._report_with(chart)), {})
